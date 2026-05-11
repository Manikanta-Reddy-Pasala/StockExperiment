# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 583.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 97 |
| ALERT2 | 95 |
| ALERT2_SKIP | 48 |
| ALERT3 | 300 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 124 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 126 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 83
- **Target hits / Stop hits / Partials:** 8 / 126 / 15
- **Avg / median % per leg:** 0.65% / -0.45%
- **Sum % (uncompounded):** 96.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 20 | 31.2% | 7 | 57 | 0 | 0.21% | 13.4% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 0 | 6 | 0 | 0.85% | 5.1% |
| BUY @ 3rd Alert (retest2) | 58 | 15 | 25.9% | 7 | 51 | 0 | 0.14% | 8.3% |
| SELL (all) | 85 | 46 | 54.1% | 1 | 69 | 15 | 0.98% | 83.5% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | -0.65% | -2.6% |
| SELL @ 3rd Alert (retest2) | 81 | 43 | 53.1% | 1 | 65 | 15 | 1.06% | 86.1% |
| retest1 (combined) | 10 | 8 | 80.0% | 0 | 10 | 0 | 0.25% | 2.5% |
| retest2 (combined) | 139 | 58 | 41.7% | 8 | 116 | 15 | 0.68% | 94.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 441.55 | 434.31 | 433.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 444.75 | 436.40 | 434.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 446.60 | 447.20 | 443.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:30:00 | 448.00 | 447.20 | 443.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 463.55 | 456.66 | 452.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 450.45 | 456.66 | 452.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 452.20 | 459.57 | 457.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 451.50 | 459.57 | 457.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 450.05 | 457.66 | 456.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 449.50 | 457.66 | 456.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 448.00 | 455.73 | 455.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 445.25 | 452.41 | 454.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 442.55 | 439.76 | 443.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 442.55 | 439.76 | 443.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 442.55 | 439.76 | 443.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 442.55 | 439.76 | 443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 441.85 | 438.90 | 441.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 441.85 | 438.90 | 441.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 441.90 | 439.50 | 441.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 442.80 | 439.50 | 441.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 443.35 | 440.27 | 441.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 443.35 | 440.27 | 441.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 440.00 | 440.22 | 441.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:45:00 | 441.10 | 440.22 | 441.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 440.00 | 440.17 | 441.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 439.05 | 440.17 | 441.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 438.10 | 439.76 | 441.06 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 448.50 | 443.02 | 442.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 450.00 | 444.41 | 443.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 454.35 | 455.41 | 452.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 454.35 | 455.41 | 452.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 454.35 | 455.41 | 452.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 454.35 | 455.41 | 452.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 454.00 | 455.12 | 453.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 451.60 | 454.30 | 452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 455.05 | 454.45 | 453.06 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 448.80 | 452.05 | 452.39 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 456.50 | 452.94 | 452.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 10:15:00 | 472.00 | 456.75 | 454.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 13:15:00 | 457.80 | 459.38 | 456.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 457.80 | 459.38 | 456.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 452.05 | 457.92 | 456.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 452.60 | 457.92 | 456.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 451.30 | 456.59 | 455.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 454.55 | 456.59 | 455.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 10:00:00 | 455.00 | 456.28 | 455.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 442.05 | 454.12 | 455.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 442.05 | 454.12 | 455.14 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 463.60 | 453.34 | 452.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 475.65 | 460.64 | 456.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 473.95 | 474.15 | 470.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:45:00 | 474.30 | 474.15 | 470.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 471.90 | 473.53 | 470.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 471.90 | 473.53 | 470.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 469.65 | 472.76 | 470.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 469.65 | 472.76 | 470.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 466.10 | 471.43 | 470.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:45:00 | 470.10 | 470.63 | 470.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 10:15:00 | 464.55 | 469.41 | 469.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 464.55 | 469.41 | 469.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 462.75 | 466.42 | 468.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 09:15:00 | 466.00 | 465.64 | 467.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 466.00 | 465.64 | 467.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 466.00 | 465.64 | 467.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 468.40 | 465.64 | 467.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 473.95 | 467.30 | 467.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 473.95 | 467.30 | 467.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 485.50 | 470.94 | 469.52 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 471.05 | 474.68 | 474.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 467.00 | 472.71 | 473.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 470.85 | 470.66 | 472.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 12:00:00 | 470.85 | 470.66 | 472.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 476.00 | 471.73 | 472.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:30:00 | 473.10 | 471.73 | 472.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 479.75 | 473.33 | 473.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:00:00 | 479.75 | 473.33 | 473.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 495.50 | 477.77 | 475.41 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 478.75 | 480.65 | 480.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 476.25 | 479.37 | 480.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 484.30 | 480.36 | 480.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 484.30 | 480.36 | 480.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 484.30 | 480.36 | 480.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 484.30 | 480.36 | 480.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 475.85 | 479.46 | 480.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 474.20 | 479.46 | 480.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:45:00 | 474.80 | 477.66 | 478.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:30:00 | 475.00 | 478.42 | 478.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:45:00 | 475.00 | 478.94 | 479.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 477.45 | 478.64 | 478.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 485.00 | 479.69 | 479.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 485.00 | 479.69 | 479.29 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 15:15:00 | 477.60 | 482.60 | 482.78 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 486.40 | 482.69 | 482.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 496.00 | 485.35 | 483.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 488.90 | 490.07 | 487.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 10:15:00 | 488.90 | 490.07 | 487.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 488.90 | 490.07 | 487.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 488.75 | 490.07 | 487.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 485.50 | 489.15 | 487.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 485.50 | 489.15 | 487.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 484.50 | 488.22 | 487.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 484.50 | 488.22 | 487.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 484.00 | 487.05 | 486.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 484.00 | 487.05 | 486.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 485.00 | 486.64 | 486.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 486.35 | 486.64 | 486.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:00:00 | 486.85 | 486.68 | 486.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 10:15:00 | 485.00 | 486.35 | 486.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 10:15:00 | 485.00 | 486.35 | 486.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 482.65 | 484.75 | 485.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 11:15:00 | 488.25 | 484.56 | 485.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 11:15:00 | 488.25 | 484.56 | 485.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 488.25 | 484.56 | 485.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 488.25 | 484.56 | 485.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 489.15 | 485.48 | 485.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:45:00 | 489.15 | 485.48 | 485.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 485.30 | 485.44 | 485.47 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 485.85 | 485.52 | 485.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 10:15:00 | 488.10 | 486.11 | 485.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 517.00 | 518.23 | 509.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 517.00 | 518.23 | 509.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 513.80 | 516.67 | 511.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 512.95 | 516.67 | 511.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 510.15 | 515.37 | 511.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:30:00 | 511.10 | 515.37 | 511.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 506.65 | 513.62 | 511.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 506.65 | 513.62 | 511.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 528.40 | 516.53 | 513.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 511.80 | 516.53 | 513.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 523.55 | 525.96 | 521.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:15:00 | 527.15 | 524.69 | 522.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 516.10 | 522.52 | 521.80 | SL hit (close<static) qty=1.00 sl=521.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 528.10 | 544.32 | 545.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 526.85 | 540.83 | 543.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 533.45 | 531.10 | 536.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 533.45 | 531.10 | 536.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 535.45 | 531.97 | 536.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 535.45 | 531.97 | 536.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 537.15 | 533.01 | 536.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 537.15 | 533.01 | 536.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 537.60 | 533.93 | 536.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 537.65 | 533.93 | 536.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 537.85 | 534.71 | 536.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 537.15 | 534.71 | 536.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 532.35 | 534.24 | 536.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 537.25 | 534.24 | 536.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 532.95 | 533.90 | 535.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 518.00 | 535.41 | 536.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:30:00 | 530.25 | 531.61 | 533.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 538.35 | 533.06 | 533.43 | SL hit (close>static) qty=1.00 sl=537.80 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 537.55 | 533.95 | 533.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 11:15:00 | 557.20 | 539.48 | 536.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 564.65 | 568.60 | 563.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 564.65 | 568.60 | 563.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 564.65 | 568.60 | 563.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 564.65 | 568.60 | 563.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 556.45 | 566.17 | 563.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 556.45 | 566.17 | 563.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 556.00 | 564.13 | 562.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 558.00 | 564.13 | 562.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 553.15 | 561.96 | 561.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 553.15 | 561.96 | 561.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 554.20 | 560.41 | 561.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 548.45 | 556.93 | 559.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 548.05 | 514.86 | 523.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 548.05 | 514.86 | 523.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 548.05 | 514.86 | 523.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 548.05 | 514.86 | 523.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 549.20 | 521.73 | 525.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:15:00 | 556.65 | 521.73 | 525.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 565.95 | 530.57 | 529.30 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 517.55 | 528.64 | 528.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 515.65 | 520.46 | 523.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 11:15:00 | 516.55 | 514.16 | 518.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 11:15:00 | 516.55 | 514.16 | 518.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 516.55 | 514.16 | 518.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 516.55 | 514.16 | 518.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 519.15 | 516.07 | 518.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 519.15 | 516.07 | 518.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 519.45 | 516.75 | 518.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 508.35 | 516.75 | 518.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 09:15:00 | 482.93 | 488.90 | 491.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-21 15:15:00 | 486.00 | 485.22 | 488.18 | SL hit (close>ema200) qty=0.50 sl=485.22 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 09:15:00 | 516.55 | 492.65 | 490.40 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 502.80 | 503.87 | 503.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 10:15:00 | 500.15 | 503.12 | 503.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 491.95 | 490.90 | 494.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 491.95 | 490.90 | 494.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 491.95 | 490.90 | 494.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 489.45 | 490.90 | 494.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 15:15:00 | 488.50 | 491.28 | 493.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 489.05 | 490.83 | 492.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 489.30 | 490.64 | 492.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 492.50 | 490.77 | 491.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:00:00 | 487.30 | 491.17 | 491.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 495.95 | 492.41 | 492.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 09:15:00 | 495.95 | 492.41 | 492.29 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 486.60 | 491.33 | 491.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 483.00 | 488.90 | 490.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 487.95 | 487.14 | 489.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:45:00 | 486.40 | 487.14 | 489.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 488.25 | 487.24 | 488.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 12:45:00 | 489.10 | 487.24 | 488.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 491.10 | 488.01 | 489.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 491.25 | 488.01 | 489.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 491.20 | 488.65 | 489.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 491.20 | 488.65 | 489.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 494.75 | 489.87 | 489.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 498.95 | 491.69 | 490.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 501.85 | 502.00 | 497.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:45:00 | 502.35 | 502.00 | 497.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 498.00 | 500.74 | 498.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 497.50 | 499.62 | 497.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 497.15 | 499.12 | 497.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 498.10 | 499.12 | 497.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 495.75 | 498.45 | 497.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 495.75 | 498.45 | 497.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 498.95 | 498.55 | 497.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 496.20 | 498.55 | 497.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 495.45 | 497.93 | 497.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 491.40 | 497.93 | 497.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 497.60 | 497.86 | 497.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 499.90 | 497.86 | 497.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 495.35 | 497.29 | 497.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 495.35 | 497.29 | 497.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 11:15:00 | 494.20 | 496.67 | 497.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 494.80 | 493.76 | 494.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 494.80 | 493.76 | 494.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 494.80 | 493.76 | 494.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:15:00 | 495.30 | 493.76 | 494.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 495.00 | 494.01 | 494.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 495.00 | 494.01 | 494.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 498.35 | 494.87 | 495.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:45:00 | 499.00 | 494.87 | 495.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 499.90 | 495.88 | 495.64 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 494.00 | 495.41 | 495.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 491.55 | 494.38 | 495.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 489.70 | 489.04 | 491.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 13:00:00 | 489.70 | 489.04 | 491.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 492.25 | 489.69 | 491.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 492.25 | 489.69 | 491.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 489.95 | 489.74 | 491.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:45:00 | 491.00 | 489.74 | 491.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 489.00 | 489.59 | 491.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 491.90 | 489.59 | 491.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 489.00 | 489.47 | 491.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 488.45 | 489.47 | 491.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:45:00 | 488.95 | 489.36 | 490.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:00:00 | 487.50 | 488.99 | 490.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 15:15:00 | 464.03 | 471.60 | 473.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 15:15:00 | 464.50 | 471.60 | 473.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 472.50 | 471.78 | 473.14 | SL hit (close>ema200) qty=0.50 sl=471.78 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 476.00 | 472.22 | 472.03 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 468.00 | 471.76 | 472.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 465.30 | 469.22 | 470.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 468.90 | 467.85 | 469.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 468.90 | 467.85 | 469.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 468.90 | 467.85 | 469.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 460.30 | 466.09 | 468.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 466.95 | 460.87 | 460.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 466.95 | 460.87 | 460.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 468.80 | 463.28 | 461.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 465.95 | 468.51 | 467.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 465.95 | 468.51 | 467.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 465.95 | 468.51 | 467.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 465.55 | 468.51 | 467.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 464.60 | 467.73 | 467.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 464.40 | 467.73 | 467.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 464.25 | 466.51 | 466.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 461.65 | 465.54 | 466.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 461.95 | 461.88 | 463.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 15:00:00 | 461.95 | 461.88 | 463.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 467.50 | 463.14 | 463.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 467.50 | 463.14 | 463.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 470.00 | 464.51 | 464.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 14:15:00 | 483.95 | 470.92 | 467.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 473.65 | 474.40 | 469.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 473.65 | 474.40 | 469.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 470.30 | 473.35 | 470.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 470.65 | 473.35 | 470.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 471.80 | 473.04 | 470.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 471.10 | 473.04 | 470.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 470.30 | 472.49 | 470.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 470.30 | 472.49 | 470.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 466.50 | 471.29 | 469.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:45:00 | 466.50 | 471.29 | 469.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 466.50 | 470.33 | 469.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 460.00 | 470.33 | 469.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 474.55 | 470.57 | 469.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 476.90 | 471.89 | 470.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:45:00 | 476.30 | 472.87 | 471.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 465.20 | 471.23 | 470.72 | SL hit (close<static) qty=1.00 sl=467.35 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 464.00 | 469.78 | 470.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 461.50 | 468.13 | 469.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 444.55 | 438.05 | 444.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 444.55 | 438.05 | 444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 444.55 | 438.05 | 444.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:45:00 | 444.60 | 438.05 | 444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 437.00 | 437.84 | 444.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 446.55 | 437.84 | 444.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 424.45 | 426.65 | 432.17 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 437.45 | 432.48 | 432.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 15:15:00 | 438.65 | 433.71 | 432.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 455.50 | 464.92 | 458.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 455.50 | 464.92 | 458.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 455.50 | 464.92 | 458.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 455.50 | 464.92 | 458.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 459.05 | 463.74 | 458.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 461.35 | 463.74 | 458.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 464.00 | 477.99 | 479.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 464.00 | 477.99 | 479.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 12:15:00 | 462.45 | 471.02 | 475.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 466.00 | 465.74 | 471.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 466.00 | 465.74 | 471.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 454.50 | 452.37 | 456.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:45:00 | 453.35 | 452.61 | 456.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:00:00 | 452.45 | 452.58 | 456.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:30:00 | 453.40 | 451.74 | 455.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 12:15:00 | 453.30 | 446.58 | 445.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 12:15:00 | 453.30 | 446.58 | 445.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 455.85 | 449.34 | 447.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 13:15:00 | 475.10 | 475.44 | 470.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:00:00 | 475.10 | 475.44 | 470.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 473.25 | 474.03 | 471.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 473.50 | 474.03 | 471.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 474.70 | 474.16 | 471.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 473.80 | 474.16 | 471.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 507.90 | 512.51 | 508.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 506.70 | 512.51 | 508.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 507.50 | 511.51 | 508.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 508.25 | 511.51 | 508.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 505.90 | 510.39 | 508.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 507.45 | 510.39 | 508.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 506.60 | 509.63 | 508.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:30:00 | 510.70 | 509.16 | 508.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:45:00 | 510.95 | 509.30 | 508.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 511.90 | 510.79 | 509.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 15:15:00 | 525.30 | 525.73 | 525.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 525.30 | 525.73 | 525.76 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 534.40 | 527.38 | 526.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 544.00 | 532.03 | 528.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 545.25 | 552.10 | 547.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 545.25 | 552.10 | 547.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 545.25 | 552.10 | 547.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 545.25 | 552.10 | 547.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 546.95 | 551.07 | 547.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:45:00 | 550.80 | 548.51 | 547.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 538.25 | 545.41 | 546.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 538.25 | 545.41 | 546.01 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 15:15:00 | 548.80 | 546.05 | 546.01 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 537.00 | 544.24 | 545.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 534.80 | 542.25 | 544.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 531.35 | 527.41 | 531.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 531.35 | 527.41 | 531.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 531.35 | 527.41 | 531.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 531.35 | 527.41 | 531.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 527.00 | 527.33 | 531.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 528.95 | 527.33 | 531.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 530.45 | 527.74 | 530.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 521.15 | 526.07 | 529.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 15:15:00 | 513.90 | 508.49 | 508.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 513.90 | 508.49 | 508.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 517.00 | 510.19 | 508.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 511.20 | 513.05 | 511.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 511.20 | 513.05 | 511.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 510.65 | 512.57 | 510.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 509.50 | 512.57 | 510.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 511.75 | 512.41 | 511.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 505.45 | 512.41 | 511.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 507.00 | 511.33 | 510.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 507.10 | 511.33 | 510.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 496.50 | 508.36 | 509.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 492.25 | 501.74 | 505.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 501.95 | 497.58 | 502.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 501.95 | 497.58 | 502.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 501.95 | 497.58 | 502.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 501.95 | 497.58 | 502.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 499.40 | 497.94 | 502.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 499.40 | 497.94 | 502.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 502.15 | 498.78 | 502.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 502.50 | 498.78 | 502.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 506.05 | 500.24 | 502.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 505.70 | 500.24 | 502.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 508.85 | 501.96 | 503.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 507.65 | 501.96 | 503.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 510.10 | 504.83 | 504.31 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 498.15 | 503.49 | 503.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 495.00 | 501.79 | 502.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 481.15 | 480.97 | 487.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 13:00:00 | 481.15 | 480.97 | 487.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 463.50 | 459.52 | 464.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 475.00 | 459.52 | 464.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 483.70 | 464.36 | 466.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 483.70 | 464.36 | 466.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 492.70 | 470.03 | 468.52 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 483.00 | 488.37 | 489.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 475.00 | 484.47 | 487.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 11:15:00 | 485.10 | 480.39 | 484.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 11:15:00 | 485.10 | 480.39 | 484.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 485.10 | 480.39 | 484.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 485.10 | 480.39 | 484.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 484.00 | 481.11 | 484.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:45:00 | 485.10 | 481.11 | 484.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 473.55 | 479.60 | 483.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 461.00 | 479.31 | 482.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 437.95 | 448.04 | 459.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 441.15 | 436.61 | 446.35 | SL hit (close>ema200) qty=0.50 sl=436.61 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 453.45 | 444.90 | 444.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 454.75 | 446.87 | 445.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 456.05 | 458.02 | 454.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 456.05 | 458.02 | 454.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 456.05 | 458.02 | 454.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 455.05 | 458.02 | 454.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 455.05 | 457.14 | 454.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 455.05 | 457.14 | 454.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 451.65 | 456.04 | 454.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:00:00 | 451.65 | 456.04 | 454.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 452.20 | 455.27 | 454.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:30:00 | 463.20 | 456.94 | 455.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 446.65 | 454.80 | 455.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 446.65 | 454.80 | 455.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 440.90 | 450.21 | 453.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 444.75 | 444.26 | 447.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:00:00 | 444.75 | 444.26 | 447.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 444.95 | 444.40 | 447.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:15:00 | 444.00 | 444.40 | 447.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:45:00 | 442.65 | 444.77 | 446.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 13:15:00 | 444.20 | 444.98 | 446.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 14:15:00 | 442.70 | 444.94 | 446.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 449.30 | 444.42 | 445.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 449.30 | 444.42 | 445.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 451.55 | 445.85 | 446.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-06 10:15:00 | 451.55 | 445.85 | 446.13 | SL hit (close>static) qty=1.00 sl=449.95 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 451.85 | 447.05 | 446.65 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 443.50 | 446.74 | 446.91 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 13:15:00 | 450.35 | 447.61 | 447.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 15:15:00 | 452.75 | 449.02 | 447.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 441.65 | 447.55 | 447.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 441.65 | 447.55 | 447.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 441.65 | 447.55 | 447.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 441.65 | 447.55 | 447.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 442.50 | 446.54 | 446.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 440.75 | 445.38 | 446.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 433.10 | 430.31 | 435.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 433.10 | 430.31 | 435.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 435.50 | 431.35 | 435.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 437.55 | 431.35 | 435.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 437.85 | 432.65 | 435.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 437.85 | 432.65 | 435.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 435.20 | 433.16 | 435.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 435.20 | 433.16 | 435.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 438.00 | 434.13 | 435.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 438.00 | 434.13 | 435.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 438.55 | 435.01 | 436.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 433.55 | 435.01 | 436.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 15:15:00 | 440.00 | 435.68 | 435.70 | SL hit (close>static) qty=1.00 sl=438.55 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 393.95 | 383.59 | 383.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 398.30 | 386.53 | 384.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 396.10 | 397.38 | 392.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:00:00 | 396.10 | 397.38 | 392.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 398.95 | 401.84 | 398.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 398.95 | 401.84 | 398.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 401.20 | 401.71 | 398.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 395.55 | 401.71 | 398.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 392.30 | 399.83 | 397.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 392.30 | 399.83 | 397.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 391.50 | 398.16 | 397.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 391.50 | 398.16 | 397.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 390.20 | 396.57 | 396.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 389.90 | 395.24 | 395.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 371.50 | 367.76 | 372.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 371.50 | 367.76 | 372.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 373.40 | 368.89 | 372.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 373.40 | 368.89 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 374.50 | 370.01 | 372.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 374.50 | 370.01 | 372.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 377.00 | 371.41 | 373.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 377.00 | 371.41 | 373.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 380.80 | 375.22 | 374.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 388.50 | 377.88 | 375.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 396.80 | 397.42 | 392.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 396.80 | 397.42 | 392.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 397.45 | 398.91 | 396.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 397.50 | 398.91 | 396.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 395.20 | 397.86 | 396.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 394.35 | 397.86 | 396.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 396.85 | 397.66 | 396.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 394.70 | 397.66 | 396.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 398.85 | 397.90 | 396.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:45:00 | 399.55 | 399.07 | 397.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 403.20 | 400.31 | 398.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 396.00 | 399.93 | 400.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 396.00 | 399.93 | 400.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 14:15:00 | 393.55 | 398.66 | 399.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 395.25 | 394.25 | 396.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 14:45:00 | 394.60 | 394.25 | 396.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 395.35 | 394.47 | 396.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 398.65 | 395.30 | 396.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 398.80 | 396.00 | 396.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 398.80 | 396.00 | 396.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 395.70 | 396.12 | 396.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 393.15 | 396.49 | 396.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 399.45 | 397.10 | 396.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 399.45 | 397.10 | 396.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 407.85 | 399.62 | 398.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 447.60 | 451.97 | 431.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 447.60 | 451.97 | 431.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 433.35 | 454.79 | 443.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 14:15:00 | 452.20 | 447.09 | 442.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 464.15 | 447.01 | 443.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 12:00:00 | 451.80 | 450.12 | 446.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 14:00:00 | 463.00 | 452.62 | 447.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 446.80 | 451.46 | 447.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 446.80 | 451.46 | 447.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 445.10 | 450.18 | 447.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 448.00 | 450.18 | 447.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 445.80 | 448.69 | 447.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:45:00 | 445.80 | 448.69 | 447.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 447.35 | 448.43 | 447.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 450.30 | 448.45 | 447.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 436.00 | 445.96 | 446.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 13:15:00 | 436.00 | 445.96 | 446.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 14:15:00 | 435.35 | 443.84 | 445.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 10:15:00 | 443.05 | 442.35 | 444.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 10:15:00 | 443.05 | 442.35 | 444.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 443.05 | 442.35 | 444.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:45:00 | 443.35 | 442.35 | 444.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 446.65 | 443.21 | 444.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:00:00 | 446.65 | 443.21 | 444.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 451.85 | 444.94 | 445.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:00:00 | 451.85 | 444.94 | 445.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 456.50 | 447.25 | 446.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 460.10 | 449.82 | 447.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 14:15:00 | 497.95 | 498.28 | 492.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 15:00:00 | 497.95 | 498.28 | 492.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 498.15 | 498.05 | 492.99 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 486.00 | 491.57 | 491.78 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 495.45 | 491.96 | 491.84 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 487.60 | 492.12 | 492.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 472.25 | 487.52 | 490.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 468.95 | 467.84 | 472.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:15:00 | 468.00 | 467.84 | 472.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 12:00:00 | 467.85 | 468.08 | 472.14 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 13:00:00 | 467.00 | 467.87 | 471.67 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 461.80 | 458.50 | 461.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 461.80 | 458.50 | 461.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 464.25 | 459.65 | 462.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-02 14:15:00 | 464.25 | 459.65 | 462.19 | SL hit (close>ema400) qty=1.00 sl=462.19 alert=retest1 |

### Cycle 67 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 465.90 | 462.88 | 462.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 466.60 | 463.84 | 463.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 463.15 | 463.70 | 463.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 463.15 | 463.70 | 463.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 463.15 | 463.70 | 463.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 463.15 | 463.70 | 463.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 460.70 | 463.10 | 463.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 460.70 | 463.10 | 463.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 454.85 | 461.45 | 462.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 450.55 | 459.27 | 461.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 476.95 | 461.32 | 461.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 476.95 | 461.32 | 461.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 476.95 | 461.32 | 461.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 476.95 | 461.32 | 461.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 483.25 | 465.71 | 463.71 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 478.00 | 479.00 | 479.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 475.75 | 478.35 | 478.75 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 483.60 | 478.99 | 478.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 494.65 | 484.10 | 481.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 515.85 | 516.09 | 508.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:30:00 | 517.00 | 516.09 | 508.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 506.20 | 512.49 | 509.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 506.20 | 512.49 | 509.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 506.95 | 511.38 | 509.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 507.50 | 511.38 | 509.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 507.50 | 510.23 | 508.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 502.75 | 509.18 | 508.88 | SL hit (close<static) qty=1.00 sl=504.05 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 504.85 | 508.31 | 508.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 497.05 | 504.57 | 506.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 505.00 | 504.64 | 506.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 15:15:00 | 505.00 | 504.64 | 506.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 505.00 | 504.64 | 506.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 501.25 | 504.64 | 506.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 497.85 | 503.29 | 505.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 496.30 | 503.29 | 505.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:45:00 | 496.35 | 500.96 | 503.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 494.65 | 500.96 | 503.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:15:00 | 496.75 | 500.28 | 502.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 495.00 | 492.88 | 496.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 495.00 | 492.88 | 496.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 498.00 | 493.90 | 496.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 498.75 | 493.90 | 496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 498.85 | 494.89 | 496.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 500.35 | 494.89 | 496.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 498.00 | 495.97 | 496.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:45:00 | 498.85 | 495.97 | 496.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 498.55 | 497.37 | 497.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 498.55 | 497.37 | 497.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 501.60 | 498.22 | 497.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 498.15 | 498.49 | 497.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 13:15:00 | 498.15 | 498.49 | 497.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 498.15 | 498.49 | 497.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 501.95 | 498.60 | 498.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 490.65 | 499.12 | 499.06 | SL hit (close<static) qty=1.00 sl=495.45 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 492.00 | 497.69 | 498.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 486.90 | 491.97 | 494.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 15:15:00 | 492.40 | 489.97 | 492.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 15:15:00 | 492.40 | 489.97 | 492.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 492.40 | 489.97 | 492.67 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 12:15:00 | 488.50 | 487.91 | 487.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 495.50 | 489.43 | 488.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 489.30 | 489.41 | 488.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 489.30 | 489.41 | 488.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 489.30 | 489.41 | 488.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 489.30 | 489.41 | 488.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 490.00 | 489.52 | 488.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 495.35 | 489.52 | 488.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 485.70 | 490.64 | 489.89 | SL hit (close<static) qty=1.00 sl=488.25 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 482.95 | 489.21 | 489.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 472.75 | 481.62 | 485.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 481.90 | 476.59 | 480.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 481.90 | 476.59 | 480.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 481.90 | 476.59 | 480.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 481.90 | 476.59 | 480.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 482.50 | 477.77 | 480.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 483.65 | 477.77 | 480.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 479.25 | 478.07 | 480.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 477.15 | 478.07 | 480.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:00:00 | 477.80 | 477.92 | 479.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:00:00 | 478.85 | 478.11 | 479.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 453.91 | 463.35 | 469.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 454.91 | 463.35 | 469.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 453.29 | 459.76 | 466.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 460.20 | 458.56 | 462.52 | SL hit (close>ema200) qty=0.50 sl=458.56 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 468.45 | 463.54 | 462.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 474.15 | 465.66 | 463.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 485.85 | 486.17 | 479.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:30:00 | 492.55 | 488.01 | 484.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 12:45:00 | 493.00 | 489.06 | 485.13 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 14:00:00 | 493.30 | 489.91 | 485.88 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 493.30 | 490.58 | 486.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 488.10 | 490.58 | 486.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 497.00 | 499.99 | 498.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 497.00 | 499.99 | 498.10 | SL hit (close<ema400) qty=1.00 sl=498.10 alert=retest1 |

### Cycle 78 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 496.90 | 499.22 | 499.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 495.45 | 498.47 | 499.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 499.80 | 498.16 | 498.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 499.80 | 498.16 | 498.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 499.80 | 498.16 | 498.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 499.80 | 498.16 | 498.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 498.40 | 498.20 | 498.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 529.90 | 498.20 | 498.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 528.75 | 504.31 | 501.50 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 487.75 | 503.93 | 504.15 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 511.40 | 504.17 | 503.85 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 499.00 | 504.59 | 505.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 496.00 | 500.70 | 502.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 493.80 | 493.67 | 497.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 13:15:00 | 493.95 | 493.45 | 495.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 493.95 | 493.45 | 495.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 496.25 | 493.45 | 495.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 501.00 | 494.96 | 496.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 501.05 | 494.96 | 496.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 500.00 | 495.97 | 496.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 502.00 | 495.97 | 496.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 503.15 | 497.40 | 497.30 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 498.05 | 500.16 | 500.40 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 502.10 | 500.48 | 500.45 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 498.00 | 499.98 | 500.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 491.10 | 497.85 | 499.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 485.00 | 484.93 | 489.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 485.00 | 484.93 | 489.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 489.50 | 485.60 | 489.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 493.50 | 485.60 | 489.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 489.35 | 486.35 | 489.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 489.35 | 486.35 | 489.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 489.50 | 486.98 | 489.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 489.10 | 486.98 | 489.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 489.55 | 487.49 | 489.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:15:00 | 492.60 | 487.49 | 489.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 492.25 | 488.45 | 489.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 491.70 | 488.45 | 489.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 493.10 | 489.38 | 489.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 493.10 | 489.38 | 489.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 491.00 | 489.70 | 489.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 484.10 | 489.70 | 489.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 459.89 | 468.82 | 476.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 463.75 | 462.39 | 468.05 | SL hit (close>ema200) qty=0.50 sl=462.39 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 400.55 | 396.13 | 396.12 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 394.00 | 395.85 | 396.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 12:15:00 | 393.10 | 394.96 | 395.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 14:15:00 | 397.40 | 395.24 | 395.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 397.40 | 395.24 | 395.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 397.40 | 395.24 | 395.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 399.90 | 395.24 | 395.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 400.00 | 396.19 | 395.99 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 392.60 | 395.91 | 396.09 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 401.20 | 396.59 | 396.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 406.35 | 398.54 | 397.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 429.20 | 429.78 | 419.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 425.00 | 427.65 | 423.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 425.00 | 427.65 | 423.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 423.60 | 427.65 | 423.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 423.75 | 426.87 | 423.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 423.50 | 426.87 | 423.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 423.10 | 426.12 | 423.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 423.10 | 426.12 | 423.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 420.70 | 424.20 | 423.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 420.70 | 424.20 | 423.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 422.50 | 423.86 | 423.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 419.60 | 423.86 | 423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 418.50 | 422.79 | 422.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 413.90 | 419.56 | 421.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 15:15:00 | 415.00 | 414.31 | 416.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:15:00 | 407.10 | 414.31 | 416.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 426.55 | 407.54 | 409.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 426.55 | 407.54 | 409.63 | SL hit (close>ema400) qty=1.00 sl=409.63 alert=retest1 |

### Cycle 93 — BUY (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 12:15:00 | 431.20 | 412.27 | 411.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 13:15:00 | 442.30 | 418.28 | 414.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 14:15:00 | 422.30 | 429.96 | 424.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 14:15:00 | 422.30 | 429.96 | 424.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 422.30 | 429.96 | 424.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 422.30 | 429.96 | 424.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 424.00 | 428.77 | 424.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 412.80 | 428.77 | 424.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 413.05 | 425.63 | 423.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 413.05 | 425.63 | 423.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 11:15:00 | 417.20 | 421.72 | 422.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 409.75 | 412.54 | 414.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 412.15 | 410.48 | 412.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 412.15 | 410.48 | 412.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 412.15 | 410.48 | 412.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 412.15 | 410.48 | 412.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 410.75 | 410.54 | 412.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:30:00 | 409.60 | 410.58 | 412.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 409.75 | 410.55 | 411.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 409.70 | 410.57 | 411.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 409.65 | 410.41 | 411.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 408.65 | 410.06 | 410.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 435.00 | 410.06 | 410.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 434.30 | 414.91 | 413.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 434.30 | 414.91 | 413.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 446.00 | 430.67 | 422.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 433.25 | 433.46 | 426.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:30:00 | 433.00 | 433.46 | 426.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 427.65 | 432.82 | 428.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:15:00 | 426.80 | 432.82 | 428.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 427.20 | 431.70 | 428.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 426.80 | 431.70 | 428.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 430.60 | 430.18 | 428.69 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 11:15:00 | 425.20 | 428.18 | 428.24 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 435.40 | 428.97 | 428.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 443.15 | 436.21 | 432.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 439.60 | 440.78 | 437.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 439.60 | 440.78 | 437.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 437.35 | 439.69 | 437.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 437.45 | 439.69 | 437.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 435.85 | 438.92 | 437.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 435.85 | 438.92 | 437.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 437.40 | 438.62 | 437.27 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 434.10 | 436.30 | 436.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 431.50 | 435.34 | 436.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 419.70 | 419.50 | 423.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 419.70 | 419.50 | 423.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 412.45 | 415.57 | 418.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 411.00 | 414.66 | 418.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 409.00 | 414.60 | 416.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 410.40 | 413.68 | 414.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:45:00 | 408.40 | 411.83 | 413.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 410.00 | 411.03 | 412.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 414.30 | 411.73 | 412.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 414.85 | 412.36 | 413.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:15:00 | 415.40 | 412.36 | 413.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 411.70 | 412.30 | 413.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:45:00 | 412.90 | 412.30 | 413.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 410.70 | 411.42 | 412.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 410.70 | 411.42 | 412.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 412.85 | 411.71 | 412.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 410.95 | 411.71 | 412.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 410.95 | 411.56 | 412.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 408.25 | 411.15 | 412.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 409.25 | 410.11 | 411.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 407.20 | 408.12 | 409.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:30:00 | 406.80 | 407.30 | 408.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 402.20 | 401.04 | 400.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 402.20 | 401.04 | 400.91 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 398.35 | 400.60 | 400.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 397.75 | 399.58 | 400.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 395.10 | 393.42 | 396.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 395.10 | 393.42 | 396.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 395.10 | 393.42 | 396.16 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 403.75 | 397.16 | 397.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 407.00 | 402.41 | 400.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 405.90 | 405.96 | 403.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 12:00:00 | 406.55 | 406.08 | 403.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 404.30 | 405.72 | 403.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 404.30 | 405.72 | 403.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 404.05 | 405.39 | 403.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 403.50 | 405.39 | 403.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 405.50 | 405.41 | 403.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 403.60 | 405.05 | 403.89 | SL hit (close<ema400) qty=1.00 sl=403.89 alert=retest1 |

### Cycle 102 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 433.65 | 437.55 | 437.80 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 439.75 | 437.99 | 437.98 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 436.60 | 437.71 | 437.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 435.30 | 437.23 | 437.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 438.95 | 437.57 | 437.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 15:15:00 | 438.95 | 437.57 | 437.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 438.95 | 437.57 | 437.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 439.15 | 437.57 | 437.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 439.10 | 437.88 | 437.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 446.50 | 439.60 | 438.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 442.15 | 442.58 | 440.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 440.95 | 442.58 | 440.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 440.45 | 442.16 | 440.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 439.60 | 442.16 | 440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 440.15 | 441.75 | 440.71 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 438.85 | 439.95 | 440.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 430.60 | 437.61 | 438.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 432.20 | 430.57 | 433.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 432.20 | 430.57 | 433.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 435.30 | 431.52 | 433.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 435.30 | 431.52 | 433.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 438.45 | 432.91 | 433.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 438.45 | 432.91 | 433.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 434.50 | 433.91 | 434.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 434.50 | 433.91 | 434.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 434.20 | 433.96 | 434.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 434.20 | 433.96 | 434.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 434.50 | 434.07 | 434.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 434.15 | 434.07 | 434.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 435.00 | 434.26 | 434.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 436.25 | 434.66 | 434.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 446.25 | 446.32 | 443.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:15:00 | 449.75 | 446.32 | 443.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:00:00 | 449.30 | 446.92 | 443.60 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 449.20 | 450.00 | 447.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 449.20 | 450.00 | 447.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 448.60 | 449.66 | 447.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 448.65 | 449.66 | 447.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 451.30 | 449.99 | 448.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 452.60 | 449.99 | 448.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 455.10 | 455.83 | 453.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 454.10 | 455.83 | 453.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 455.45 | 455.80 | 454.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 457.15 | 455.80 | 454.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 464.10 | 457.46 | 455.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 457.05 | 459.69 | 457.81 | SL hit (close<ema400) qty=1.00 sl=457.81 alert=retest1 |

### Cycle 108 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 451.55 | 458.01 | 458.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 449.00 | 456.21 | 457.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 442.15 | 432.22 | 438.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 442.15 | 432.22 | 438.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 442.15 | 432.22 | 438.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 442.15 | 432.22 | 438.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 435.65 | 432.90 | 438.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:45:00 | 432.10 | 432.89 | 437.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:15:00 | 434.65 | 430.22 | 432.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 434.85 | 432.09 | 432.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 15:15:00 | 436.00 | 433.56 | 433.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 436.00 | 433.56 | 433.23 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 428.95 | 432.58 | 432.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 427.40 | 430.05 | 430.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 430.10 | 429.44 | 430.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 430.10 | 429.44 | 430.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 430.95 | 429.74 | 430.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 430.95 | 429.74 | 430.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 427.95 | 429.38 | 430.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 432.70 | 430.05 | 430.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 431.80 | 430.40 | 430.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 433.45 | 430.40 | 430.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 431.65 | 430.83 | 430.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 432.30 | 431.13 | 430.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 431.20 | 431.59 | 431.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 431.20 | 431.59 | 431.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 431.20 | 431.59 | 431.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 431.30 | 431.59 | 431.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 429.50 | 431.17 | 431.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 429.15 | 431.17 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 432.95 | 431.52 | 431.21 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 429.70 | 431.20 | 431.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 426.65 | 430.07 | 430.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 432.10 | 430.06 | 430.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 432.10 | 430.06 | 430.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 432.10 | 430.06 | 430.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 432.10 | 430.06 | 430.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 431.55 | 430.36 | 430.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 431.90 | 430.36 | 430.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 438.30 | 431.95 | 431.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 446.80 | 436.21 | 433.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 443.80 | 445.09 | 442.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:00:00 | 443.80 | 445.09 | 442.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 444.20 | 445.43 | 443.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 442.20 | 445.43 | 443.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 441.60 | 444.66 | 443.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 441.60 | 444.66 | 443.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 442.25 | 444.18 | 443.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 443.00 | 444.18 | 443.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 445.10 | 444.12 | 443.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 14:15:00 | 446.70 | 444.12 | 443.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 442.65 | 445.12 | 445.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 442.65 | 445.12 | 445.27 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 14:15:00 | 452.35 | 446.47 | 445.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 453.85 | 449.39 | 448.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 450.45 | 450.50 | 449.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 12:15:00 | 450.45 | 450.50 | 449.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 450.45 | 450.50 | 449.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 450.45 | 450.50 | 449.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 449.10 | 450.20 | 449.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 448.05 | 450.20 | 449.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 449.70 | 450.10 | 449.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 447.15 | 450.10 | 449.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 449.30 | 449.94 | 449.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 446.45 | 449.94 | 449.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 445.15 | 448.98 | 448.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 445.15 | 448.98 | 448.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 11:15:00 | 447.05 | 448.60 | 448.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 443.80 | 446.65 | 447.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 446.80 | 446.26 | 447.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 446.80 | 446.26 | 447.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 444.20 | 445.17 | 446.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 445.75 | 445.17 | 446.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 444.00 | 442.72 | 444.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 444.00 | 442.72 | 444.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 444.45 | 443.07 | 444.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 444.45 | 443.07 | 444.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 447.30 | 443.91 | 444.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 447.30 | 443.91 | 444.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 446.00 | 444.33 | 444.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 442.85 | 444.33 | 444.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 442.50 | 443.78 | 444.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 420.71 | 424.92 | 427.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 420.38 | 424.92 | 427.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 11:15:00 | 398.57 | 408.43 | 414.45 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 417.35 | 415.47 | 415.45 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 13:15:00 | 412.35 | 414.85 | 415.17 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 419.70 | 415.94 | 415.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 423.40 | 418.87 | 417.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 418.30 | 419.37 | 417.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 418.30 | 419.37 | 417.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 418.30 | 419.37 | 417.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 417.10 | 419.37 | 417.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 417.75 | 419.05 | 417.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 417.75 | 419.05 | 417.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 417.45 | 418.73 | 417.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 417.90 | 418.56 | 417.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 417.95 | 418.44 | 417.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 414.90 | 417.65 | 417.63 | SL hit (close<static) qty=1.00 sl=415.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 411.25 | 416.37 | 417.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 410.35 | 415.17 | 416.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 403.40 | 401.10 | 404.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 14:15:00 | 403.40 | 401.10 | 404.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 403.40 | 401.10 | 404.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:30:00 | 407.75 | 401.10 | 404.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 415.50 | 403.16 | 405.19 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 411.65 | 406.43 | 406.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 415.40 | 408.23 | 407.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 410.90 | 411.76 | 409.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 09:30:00 | 410.00 | 411.76 | 409.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 410.60 | 411.53 | 409.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 409.90 | 411.53 | 409.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 410.85 | 411.39 | 409.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 411.50 | 411.39 | 409.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 409.25 | 410.96 | 409.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 409.25 | 410.96 | 409.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 405.65 | 409.90 | 409.31 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 403.85 | 408.69 | 408.82 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 413.10 | 408.80 | 408.79 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 410.05 | 411.15 | 411.17 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 413.35 | 411.59 | 411.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 435.25 | 416.63 | 413.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 14:15:00 | 432.40 | 435.47 | 425.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 432.40 | 435.47 | 425.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 447.65 | 441.85 | 432.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 440.05 | 441.85 | 432.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 436.75 | 444.74 | 437.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 436.05 | 444.74 | 437.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 441.65 | 444.12 | 437.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 448.00 | 444.90 | 438.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 448.65 | 446.32 | 440.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 492.80 | 462.05 | 449.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 481.80 | 484.84 | 485.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 15:15:00 | 477.00 | 482.84 | 484.15 | Break + close below crossover candle low |

### Cycle 127 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 494.30 | 485.13 | 485.07 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 482.35 | 489.30 | 489.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 477.60 | 486.30 | 487.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 482.45 | 478.34 | 481.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 482.45 | 478.34 | 481.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 482.45 | 478.34 | 481.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 482.45 | 478.34 | 481.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 485.05 | 479.68 | 481.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 484.30 | 479.68 | 481.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 482.00 | 480.14 | 481.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 476.85 | 480.14 | 481.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 496.40 | 482.67 | 482.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 496.40 | 482.67 | 482.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 522.75 | 490.69 | 485.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 14:15:00 | 506.00 | 507.82 | 503.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:45:00 | 505.25 | 507.82 | 503.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 507.00 | 507.65 | 503.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 502.70 | 506.66 | 503.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 500.85 | 505.50 | 503.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 500.85 | 505.50 | 503.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 504.60 | 504.41 | 503.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:45:00 | 502.30 | 504.41 | 503.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 505.50 | 506.31 | 504.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 500.15 | 506.31 | 504.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 521.90 | 529.51 | 525.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 510.10 | 529.51 | 525.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 524.75 | 528.56 | 525.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 537.75 | 530.42 | 526.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 536.40 | 535.45 | 530.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 534.70 | 535.10 | 531.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 537.15 | 535.34 | 531.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 535.50 | 538.11 | 534.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 535.50 | 538.11 | 534.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 533.05 | 537.10 | 534.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 533.05 | 537.10 | 534.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 544.10 | 538.50 | 535.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 549.80 | 541.96 | 537.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 532.60 | 537.97 | 538.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 532.60 | 537.97 | 538.34 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 13:15:00 | 544.00 | 539.34 | 538.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 14:15:00 | 547.70 | 541.01 | 539.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 535.85 | 540.72 | 539.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 535.85 | 540.72 | 539.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 535.85 | 540.72 | 539.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:30:00 | 531.25 | 540.72 | 539.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 532.50 | 539.08 | 539.18 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 13:15:00 | 543.35 | 539.74 | 539.41 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 532.00 | 537.83 | 538.57 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 546.00 | 540.08 | 539.49 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 520.20 | 536.95 | 538.93 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 540.65 | 537.40 | 537.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 542.65 | 538.45 | 537.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 539.75 | 540.13 | 538.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 539.75 | 540.13 | 538.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 538.75 | 539.86 | 538.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 538.30 | 539.86 | 538.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 544.40 | 540.76 | 539.40 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 536.35 | 539.71 | 540.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 534.30 | 538.63 | 539.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 531.75 | 531.75 | 534.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 531.20 | 531.75 | 534.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 551.90 | 535.90 | 536.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 559.50 | 535.90 | 536.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 552.80 | 539.28 | 537.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 556.10 | 550.56 | 545.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 13:15:00 | 549.95 | 551.67 | 547.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 14:00:00 | 549.95 | 551.67 | 547.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 547.75 | 550.88 | 547.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:30:00 | 545.80 | 550.88 | 547.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 547.50 | 550.21 | 547.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 540.25 | 550.21 | 547.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 542.25 | 548.62 | 546.84 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 540.75 | 545.03 | 545.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 536.00 | 543.23 | 544.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 534.80 | 534.08 | 538.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:45:00 | 534.10 | 534.08 | 538.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 538.15 | 534.89 | 538.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 538.15 | 534.89 | 538.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 537.05 | 535.32 | 538.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 539.00 | 535.32 | 538.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 534.35 | 535.13 | 538.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 534.35 | 535.13 | 538.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 527.95 | 533.98 | 537.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 13:00:00 | 524.05 | 531.22 | 535.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 538.35 | 535.17 | 535.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 538.35 | 535.17 | 535.15 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 13:15:00 | 532.00 | 535.03 | 535.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 528.85 | 533.80 | 534.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 523.70 | 521.22 | 526.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:45:00 | 522.40 | 521.22 | 526.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 531.00 | 524.01 | 526.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 518.30 | 524.01 | 526.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 11:45:00 | 517.75 | 520.40 | 524.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 15:15:00 | 521.00 | 521.51 | 523.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 531.70 | 525.81 | 525.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 531.70 | 525.81 | 525.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 11:15:00 | 534.55 | 529.08 | 527.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 528.30 | 531.86 | 529.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 528.30 | 531.86 | 529.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 528.30 | 531.86 | 529.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 535.60 | 532.09 | 530.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 535.15 | 532.72 | 530.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 537.70 | 533.47 | 531.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:45:00 | 536.10 | 535.07 | 532.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 551.50 | 553.95 | 549.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 549.65 | 553.95 | 549.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 549.50 | 553.06 | 549.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 556.85 | 553.36 | 549.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 551.70 | 556.46 | 553.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 551.75 | 554.96 | 553.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 547.60 | 552.28 | 552.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 547.60 | 552.28 | 552.45 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 553.85 | 552.56 | 552.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 556.50 | 553.29 | 552.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 549.80 | 553.34 | 552.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 549.80 | 553.34 | 552.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 549.80 | 553.34 | 552.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 549.80 | 553.34 | 552.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 550.15 | 552.70 | 552.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 550.00 | 552.70 | 552.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 545.80 | 551.32 | 552.06 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 559.00 | 550.55 | 550.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 571.50 | 554.75 | 552.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 560.30 | 561.59 | 557.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 560.30 | 561.59 | 557.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 562.35 | 561.45 | 558.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:30:00 | 564.30 | 563.05 | 559.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 09:15:00 | 620.73 | 611.88 | 608.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 609.00 | 619.72 | 619.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 15:15:00 | 607.00 | 615.63 | 617.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 617.00 | 615.90 | 617.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 617.00 | 615.90 | 617.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 617.00 | 615.90 | 617.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:15:00 | 617.15 | 615.90 | 617.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 619.75 | 616.67 | 617.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:45:00 | 618.65 | 616.67 | 617.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 621.10 | 617.56 | 618.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 621.10 | 617.56 | 618.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:15:00 | 426.00 | 2024-05-13 11:15:00 | 441.55 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-06-03 09:15:00 | 454.55 | 2024-06-04 10:15:00 | 442.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-06-03 10:00:00 | 455.00 | 2024-06-04 10:15:00 | 442.05 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-06-11 09:45:00 | 470.10 | 2024-06-11 10:15:00 | 464.55 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-06-21 15:15:00 | 474.20 | 2024-06-26 09:15:00 | 485.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-06-25 09:45:00 | 474.80 | 2024-06-26 09:15:00 | 485.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-06-25 10:30:00 | 475.00 | 2024-06-26 09:15:00 | 485.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-06-25 13:45:00 | 475.00 | 2024-06-26 09:15:00 | 485.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-07-03 09:15:00 | 486.35 | 2024-07-03 10:15:00 | 485.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-07-03 10:00:00 | 486.85 | 2024-07-03 10:15:00 | 485.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-12 10:15:00 | 527.15 | 2024-07-12 11:15:00 | 516.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-07-12 12:15:00 | 526.80 | 2024-07-12 15:15:00 | 520.85 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-15 09:15:00 | 529.45 | 2024-07-15 09:15:00 | 582.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-23 12:15:00 | 518.00 | 2024-07-24 13:15:00 | 538.35 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2024-07-24 11:30:00 | 530.25 | 2024-07-24 13:15:00 | 538.35 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-12 09:15:00 | 508.35 | 2024-08-21 09:15:00 | 482.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-12 09:15:00 | 508.35 | 2024-08-21 15:15:00 | 486.00 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2024-09-03 10:15:00 | 489.45 | 2024-09-06 09:15:00 | 495.95 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-09-03 15:15:00 | 488.50 | 2024-09-06 09:15:00 | 495.95 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-09-04 10:30:00 | 489.05 | 2024-09-06 09:15:00 | 495.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-04 14:45:00 | 489.30 | 2024-09-06 09:15:00 | 495.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-09-05 15:00:00 | 487.30 | 2024-09-06 09:15:00 | 495.95 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-09-12 15:15:00 | 499.90 | 2024-09-13 10:15:00 | 495.35 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-20 10:15:00 | 488.45 | 2024-09-26 15:15:00 | 464.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 10:45:00 | 488.95 | 2024-09-26 15:15:00 | 464.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 10:15:00 | 488.45 | 2024-09-27 09:15:00 | 472.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2024-09-20 10:45:00 | 488.95 | 2024-09-27 09:15:00 | 472.50 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2024-09-20 12:00:00 | 487.50 | 2024-09-30 12:15:00 | 476.00 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2024-10-07 09:30:00 | 460.30 | 2024-10-09 10:15:00 | 466.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-18 14:15:00 | 476.90 | 2024-10-21 09:15:00 | 465.20 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-10-18 14:45:00 | 476.30 | 2024-10-21 09:15:00 | 465.20 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-11-04 11:15:00 | 461.35 | 2024-11-13 09:15:00 | 464.00 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-11-19 11:45:00 | 453.35 | 2024-11-25 12:15:00 | 453.30 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-11-19 13:00:00 | 452.45 | 2024-11-25 12:15:00 | 453.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-11-19 14:30:00 | 453.40 | 2024-11-25 12:15:00 | 453.30 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-12-06 12:30:00 | 510.70 | 2024-12-12 15:15:00 | 525.30 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2024-12-06 13:45:00 | 510.95 | 2024-12-12 15:15:00 | 525.30 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2024-12-09 09:30:00 | 511.90 | 2024-12-12 15:15:00 | 525.30 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2024-12-18 13:45:00 | 550.80 | 2024-12-19 09:15:00 | 538.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-12-26 09:30:00 | 521.15 | 2025-01-02 15:15:00 | 513.90 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-01-23 09:15:00 | 461.00 | 2025-01-24 14:15:00 | 437.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 461.00 | 2025-01-27 14:15:00 | 441.15 | STOP_HIT | 0.50 | 4.31% |
| BUY | retest2 | 2025-02-01 09:30:00 | 463.20 | 2025-02-03 09:15:00 | 446.65 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-02-04 14:15:00 | 444.00 | 2025-02-06 10:15:00 | 451.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-02-05 11:45:00 | 442.65 | 2025-02-06 10:15:00 | 451.55 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-02-05 13:15:00 | 444.20 | 2025-02-06 10:15:00 | 451.55 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-05 14:15:00 | 442.70 | 2025-02-06 10:15:00 | 451.55 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 433.55 | 2025-02-13 15:15:00 | 440.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-02-14 09:15:00 | 432.20 | 2025-02-24 09:15:00 | 414.77 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-02-14 10:30:00 | 436.60 | 2025-02-24 09:15:00 | 414.10 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-02-14 11:45:00 | 435.90 | 2025-02-24 12:15:00 | 410.59 | PARTIAL | 0.50 | 5.81% |
| SELL | retest2 | 2025-02-17 09:15:00 | 427.35 | 2025-02-24 14:15:00 | 410.02 | PARTIAL | 0.50 | 4.06% |
| SELL | retest2 | 2025-02-17 13:30:00 | 431.60 | 2025-02-25 09:15:00 | 405.98 | PARTIAL | 0.50 | 5.94% |
| SELL | retest2 | 2025-02-14 09:15:00 | 432.20 | 2025-02-25 14:15:00 | 409.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2025-02-14 10:30:00 | 436.60 | 2025-02-25 14:15:00 | 409.00 | STOP_HIT | 0.50 | 6.32% |
| SELL | retest2 | 2025-02-14 11:45:00 | 435.90 | 2025-02-25 14:15:00 | 409.00 | STOP_HIT | 0.50 | 6.17% |
| SELL | retest2 | 2025-02-17 09:15:00 | 427.35 | 2025-02-25 14:15:00 | 409.00 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-02-17 13:30:00 | 431.60 | 2025-02-25 14:15:00 | 409.00 | STOP_HIT | 0.50 | 5.24% |
| BUY | retest2 | 2025-03-25 14:45:00 | 399.55 | 2025-03-27 13:15:00 | 396.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-26 09:45:00 | 403.20 | 2025-03-27 13:15:00 | 396.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-04-02 09:15:00 | 393.15 | 2025-04-02 11:15:00 | 399.45 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-04-07 14:15:00 | 452.20 | 2025-04-09 13:15:00 | 436.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-04-08 09:15:00 | 464.15 | 2025-04-09 13:15:00 | 436.00 | STOP_HIT | 1.00 | -6.06% |
| BUY | retest2 | 2025-04-08 12:00:00 | 451.80 | 2025-04-09 13:15:00 | 436.00 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2025-04-08 14:00:00 | 463.00 | 2025-04-09 13:15:00 | 436.00 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2025-04-09 12:30:00 | 450.30 | 2025-04-09 13:15:00 | 436.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest1 | 2025-04-29 10:15:00 | 468.00 | 2025-05-02 14:15:00 | 464.25 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest1 | 2025-04-29 12:00:00 | 467.85 | 2025-05-02 14:15:00 | 464.25 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest1 | 2025-04-29 13:00:00 | 467.00 | 2025-05-02 14:15:00 | 464.25 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-05-05 14:00:00 | 458.35 | 2025-05-05 14:15:00 | 465.35 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-05 14:30:00 | 457.95 | 2025-05-05 15:15:00 | 465.90 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-21 13:15:00 | 507.50 | 2025-05-22 09:15:00 | 502.75 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-21 14:15:00 | 507.50 | 2025-05-22 09:15:00 | 502.75 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-23 10:15:00 | 496.30 | 2025-05-29 10:15:00 | 498.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-26 12:45:00 | 496.35 | 2025-05-29 10:15:00 | 498.55 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-05-26 13:15:00 | 494.65 | 2025-05-29 10:15:00 | 498.55 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-05-26 14:15:00 | 496.75 | 2025-05-29 10:15:00 | 498.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-05-29 15:15:00 | 501.95 | 2025-05-30 14:15:00 | 490.65 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-11 09:15:00 | 495.35 | 2025-06-11 13:15:00 | 485.70 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-11 15:00:00 | 494.50 | 2025-06-12 13:15:00 | 487.95 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-06-12 14:15:00 | 490.50 | 2025-06-13 09:15:00 | 482.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-06-17 12:15:00 | 477.15 | 2025-06-19 13:15:00 | 453.91 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-06-17 14:00:00 | 477.80 | 2025-06-19 13:15:00 | 454.91 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-06-17 15:00:00 | 478.85 | 2025-06-20 09:15:00 | 453.29 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-06-17 12:15:00 | 477.15 | 2025-06-20 15:15:00 | 460.20 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-06-17 14:00:00 | 477.80 | 2025-06-20 15:15:00 | 460.20 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-06-17 15:00:00 | 478.85 | 2025-06-20 15:15:00 | 460.20 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest1 | 2025-06-27 11:30:00 | 492.55 | 2025-07-02 14:15:00 | 497.00 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2025-06-27 12:45:00 | 493.00 | 2025-07-02 14:15:00 | 497.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest1 | 2025-06-27 14:00:00 | 493.30 | 2025-07-02 14:15:00 | 497.00 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2025-07-03 11:15:00 | 505.85 | 2025-07-07 11:15:00 | 496.90 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-07-03 12:00:00 | 505.00 | 2025-07-07 11:15:00 | 496.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-07 09:30:00 | 505.35 | 2025-07-07 11:15:00 | 496.90 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-25 09:15:00 | 484.10 | 2025-07-28 14:15:00 | 459.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 484.10 | 2025-07-29 15:15:00 | 463.75 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest1 | 2025-08-26 09:15:00 | 407.10 | 2025-08-28 11:15:00 | 426.55 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2025-09-08 12:30:00 | 409.60 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 1.00 | -6.03% |
| SELL | retest2 | 2025-09-08 14:45:00 | 409.75 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2025-09-09 11:45:00 | 409.70 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2025-09-09 14:15:00 | 409.65 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2025-09-25 12:00:00 | 411.00 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2025-09-26 09:15:00 | 409.00 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-09-29 11:15:00 | 410.40 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2025-09-29 13:45:00 | 408.40 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-10-01 09:15:00 | 410.95 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-10-01 10:00:00 | 410.95 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-10-01 10:30:00 | 408.25 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-10-03 10:00:00 | 409.25 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2025-10-06 10:30:00 | 406.80 | 2025-10-10 14:15:00 | 402.20 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest1 | 2025-10-17 12:00:00 | 406.55 | 2025-10-17 15:15:00 | 403.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-20 12:45:00 | 406.25 | 2025-10-23 09:15:00 | 446.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 13:15:00 | 407.70 | 2025-10-23 09:15:00 | 448.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 15:15:00 | 407.25 | 2025-10-23 09:15:00 | 447.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-11-13 09:15:00 | 449.75 | 2025-11-20 09:15:00 | 457.05 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest1 | 2025-11-13 10:00:00 | 449.30 | 2025-11-20 09:15:00 | 457.05 | STOP_HIT | 1.00 | 1.72% |
| BUY | retest2 | 2025-11-19 09:15:00 | 457.15 | 2025-11-21 14:15:00 | 451.55 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-19 10:00:00 | 464.10 | 2025-11-21 14:15:00 | 451.55 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-11-20 12:00:00 | 457.70 | 2025-11-21 14:15:00 | 451.55 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-21 14:15:00 | 457.30 | 2025-11-21 14:15:00 | 451.55 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-26 11:45:00 | 432.10 | 2025-11-28 15:15:00 | 436.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-28 12:15:00 | 434.65 | 2025-11-28 15:15:00 | 436.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-28 14:15:00 | 434.85 | 2025-11-28 15:15:00 | 436.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-12-15 14:15:00 | 446.70 | 2025-12-17 11:15:00 | 442.65 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 09:15:00 | 442.85 | 2026-01-08 11:15:00 | 420.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 15:15:00 | 442.50 | 2026-01-08 11:15:00 | 420.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:15:00 | 442.85 | 2026-01-12 11:15:00 | 398.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-29 15:15:00 | 442.50 | 2026-01-12 14:15:00 | 417.85 | STOP_HIT | 0.50 | 5.57% |
| BUY | retest2 | 2026-01-16 13:00:00 | 417.90 | 2026-01-19 09:15:00 | 414.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-01-16 15:15:00 | 417.95 | 2026-01-19 09:15:00 | 414.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-02-02 12:00:00 | 448.00 | 2026-02-03 09:15:00 | 492.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 13:30:00 | 448.65 | 2026-02-03 09:15:00 | 493.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 476.85 | 2026-02-13 11:15:00 | 496.40 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-02-24 14:00:00 | 537.75 | 2026-03-02 11:15:00 | 532.60 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-25 09:30:00 | 536.40 | 2026-03-02 11:15:00 | 532.60 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-25 12:30:00 | 534.70 | 2026-03-02 11:15:00 | 532.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2026-02-25 13:45:00 | 537.15 | 2026-03-02 11:15:00 | 532.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-26 14:30:00 | 549.80 | 2026-03-02 11:15:00 | 532.60 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-03-23 13:00:00 | 524.05 | 2026-03-25 10:15:00 | 538.35 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-03-30 09:15:00 | 518.30 | 2026-04-01 12:15:00 | 531.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-03-30 11:45:00 | 517.75 | 2026-04-01 12:15:00 | 531.70 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-03-30 15:15:00 | 521.00 | 2026-04-01 12:15:00 | 531.70 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-06 11:15:00 | 535.60 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2026-04-06 11:45:00 | 535.15 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2026-04-06 14:15:00 | 537.70 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2026-04-07 09:45:00 | 536.10 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 556.85 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-04-13 10:15:00 | 551.70 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-04-13 11:45:00 | 551.75 | 2026-04-13 13:15:00 | 547.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-04-22 10:30:00 | 564.30 | 2026-05-04 09:15:00 | 620.73 | TARGET_HIT | 1.00 | 10.00% |
