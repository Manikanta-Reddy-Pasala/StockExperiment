# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 686.95
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
| ALERT2 | 51 |
| ALERT2_SKIP | 28 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 58 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 48
- **Target hits / Stop hits / Partials:** 3 / 58 / 5
- **Avg / median % per leg:** 0.41% / -0.41%
- **Sum % (uncompounded):** 27.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 4 | 15.4% | 3 | 23 | 0 | 0.49% | 12.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.64% | -4.9% |
| BUY @ 3rd Alert (retest2) | 23 | 4 | 17.4% | 3 | 20 | 0 | 0.77% | 17.7% |
| SELL (all) | 40 | 14 | 35.0% | 0 | 35 | 5 | 0.36% | 14.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 14 | 35.0% | 0 | 35 | 5 | 0.36% | 14.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.64% | -4.9% |
| retest2 (combined) | 63 | 18 | 28.6% | 3 | 55 | 5 | 0.51% | 32.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 639.35 | 623.04 | 622.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 651.95 | 632.09 | 627.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 679.55 | 679.63 | 669.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 679.55 | 679.63 | 669.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 703.50 | 701.89 | 699.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 702.00 | 701.89 | 699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 701.25 | 704.13 | 701.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 713.80 | 704.13 | 701.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 707.10 | 708.05 | 706.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 706.75 | 707.79 | 706.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 707.25 | 707.07 | 706.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 705.95 | 706.85 | 706.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 704.90 | 706.85 | 706.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 704.10 | 706.30 | 706.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 711.95 | 706.30 | 706.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 706.05 | 708.89 | 708.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:15:00 | 706.35 | 708.89 | 708.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:45:00 | 707.10 | 708.35 | 707.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 707.30 | 708.14 | 707.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 704.70 | 707.35 | 707.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 704.70 | 707.35 | 707.53 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 714.90 | 708.43 | 707.89 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 702.05 | 707.67 | 707.76 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 706.45 | 705.69 | 705.69 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 703.75 | 705.30 | 705.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 701.20 | 704.48 | 705.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 703.50 | 701.51 | 703.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 703.50 | 701.51 | 703.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 703.50 | 701.51 | 703.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 703.50 | 701.51 | 703.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 704.05 | 702.02 | 703.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 706.35 | 702.02 | 703.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 705.15 | 702.64 | 703.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 704.85 | 702.64 | 703.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 704.10 | 702.94 | 703.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 704.10 | 702.94 | 703.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 702.80 | 702.91 | 703.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 14:15:00 | 701.30 | 702.91 | 703.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 713.80 | 704.45 | 703.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 713.80 | 704.45 | 703.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 720.65 | 707.69 | 705.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 707.70 | 712.41 | 709.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 707.70 | 712.41 | 709.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 707.70 | 712.41 | 709.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 707.60 | 712.41 | 709.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 704.80 | 710.89 | 709.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 704.15 | 710.89 | 709.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 705.90 | 707.71 | 707.95 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 711.90 | 708.55 | 708.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 15:15:00 | 713.00 | 709.44 | 708.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 704.50 | 708.45 | 708.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 704.50 | 708.45 | 708.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 704.50 | 708.45 | 708.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 704.50 | 708.45 | 708.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 704.90 | 707.74 | 708.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 700.80 | 705.90 | 707.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 694.00 | 692.16 | 696.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:30:00 | 694.70 | 692.16 | 696.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 693.20 | 692.37 | 695.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 690.85 | 692.37 | 695.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 689.00 | 691.70 | 695.25 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 694.40 | 691.82 | 691.49 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 685.65 | 690.46 | 690.94 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 694.45 | 690.85 | 690.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 704.25 | 693.53 | 692.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 697.50 | 698.94 | 695.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 697.50 | 698.94 | 695.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 697.50 | 698.94 | 695.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 698.90 | 698.94 | 695.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 700.50 | 698.95 | 696.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:45:00 | 705.20 | 699.76 | 697.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 701.00 | 708.15 | 708.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 701.00 | 708.15 | 708.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 695.40 | 699.71 | 703.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 693.85 | 692.66 | 695.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 12:15:00 | 693.85 | 692.66 | 695.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 693.85 | 692.66 | 695.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 692.20 | 692.66 | 695.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 695.25 | 693.18 | 695.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:15:00 | 693.65 | 693.18 | 695.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 693.75 | 693.29 | 694.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:15:00 | 692.60 | 693.29 | 694.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 692.60 | 693.15 | 694.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 698.80 | 693.15 | 694.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 705.20 | 695.56 | 695.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 703.85 | 695.56 | 695.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 710.50 | 698.55 | 697.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 12:15:00 | 716.10 | 703.84 | 699.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 712.10 | 716.27 | 708.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 712.10 | 716.27 | 708.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 712.10 | 716.27 | 708.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 712.10 | 716.27 | 708.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 707.85 | 714.58 | 708.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 707.85 | 714.58 | 708.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 707.50 | 713.17 | 708.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 708.50 | 713.17 | 708.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 709.20 | 712.37 | 708.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 714.50 | 710.52 | 708.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 705.00 | 709.07 | 707.93 | SL hit (close<static) qty=1.00 sl=707.35 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 704.10 | 707.13 | 707.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 702.25 | 704.41 | 705.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 703.55 | 702.90 | 704.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 703.55 | 702.90 | 704.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 703.55 | 702.90 | 704.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 703.30 | 702.90 | 704.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 702.90 | 702.86 | 704.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 704.30 | 702.86 | 704.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 701.40 | 701.83 | 703.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 701.10 | 701.83 | 703.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 700.25 | 701.51 | 702.80 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 15:15:00 | 705.00 | 703.50 | 703.37 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 697.15 | 702.23 | 702.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 686.00 | 698.98 | 701.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 704.70 | 695.25 | 697.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 704.70 | 695.25 | 697.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 704.70 | 695.25 | 697.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 704.70 | 695.25 | 697.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 718.40 | 699.88 | 699.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 730.50 | 710.72 | 705.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 12:15:00 | 748.15 | 751.04 | 741.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 13:00:00 | 748.15 | 751.04 | 741.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 751.90 | 761.54 | 757.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 751.90 | 761.54 | 757.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 749.95 | 759.22 | 756.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 749.95 | 759.22 | 756.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 763.60 | 760.19 | 758.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 764.60 | 760.19 | 758.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 759.00 | 759.88 | 758.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 759.00 | 759.88 | 758.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 757.40 | 759.47 | 758.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 757.50 | 759.47 | 758.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 763.80 | 760.33 | 758.91 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 753.10 | 758.08 | 758.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 751.35 | 755.34 | 756.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 738.20 | 735.31 | 742.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:45:00 | 740.40 | 735.31 | 742.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 740.60 | 736.37 | 741.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 740.60 | 736.37 | 741.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 740.65 | 737.23 | 741.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:00:00 | 738.05 | 738.17 | 741.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 737.65 | 738.04 | 740.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 744.10 | 739.50 | 740.79 | SL hit (close>static) qty=1.00 sl=741.80 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 751.25 | 742.65 | 742.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 760.65 | 746.25 | 743.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 760.00 | 760.25 | 753.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 760.75 | 760.25 | 753.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 758.10 | 759.82 | 753.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 758.10 | 759.82 | 753.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 753.00 | 758.64 | 755.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 753.00 | 758.64 | 755.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 751.45 | 757.20 | 754.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 759.15 | 757.20 | 754.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 757.50 | 757.43 | 755.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 749.30 | 753.99 | 754.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 749.30 | 753.99 | 754.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 744.60 | 752.11 | 753.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 13:15:00 | 754.70 | 752.02 | 753.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 754.70 | 752.02 | 753.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 754.70 | 752.02 | 753.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:45:00 | 756.00 | 752.02 | 753.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 753.40 | 752.29 | 753.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:30:00 | 753.70 | 752.29 | 753.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 752.15 | 752.27 | 753.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 751.60 | 752.27 | 753.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 743.05 | 750.42 | 752.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:45:00 | 742.65 | 748.72 | 751.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:00:00 | 742.95 | 746.62 | 749.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 742.45 | 745.39 | 748.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 741.50 | 741.07 | 741.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 740.00 | 740.86 | 741.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 741.55 | 740.86 | 741.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 736.50 | 739.98 | 741.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 763.00 | 744.78 | 743.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 763.00 | 744.78 | 743.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 764.55 | 752.52 | 747.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 800.95 | 801.17 | 791.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 800.95 | 801.17 | 791.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 814.20 | 816.92 | 811.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 812.80 | 816.92 | 811.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 810.95 | 816.10 | 812.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 810.95 | 816.10 | 812.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 804.15 | 813.71 | 811.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 804.15 | 813.71 | 811.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 798.70 | 810.71 | 810.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 798.70 | 810.71 | 810.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 795.00 | 807.57 | 808.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 792.50 | 801.25 | 805.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 796.70 | 793.99 | 799.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 796.70 | 793.99 | 799.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 803.30 | 795.85 | 799.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 803.30 | 795.85 | 799.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 808.50 | 798.38 | 800.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 808.50 | 798.38 | 800.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 802.55 | 801.62 | 801.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 800.85 | 801.38 | 801.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 809.40 | 802.98 | 802.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 809.40 | 802.98 | 802.31 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 802.40 | 803.87 | 804.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 796.60 | 802.20 | 803.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 801.00 | 799.80 | 801.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 801.00 | 799.80 | 801.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 801.00 | 799.80 | 801.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 800.80 | 799.80 | 801.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 794.30 | 798.70 | 801.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:00:00 | 791.50 | 797.26 | 800.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 803.45 | 797.55 | 799.22 | SL hit (close>static) qty=1.00 sl=801.05 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 790.80 | 786.52 | 786.06 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 780.90 | 786.23 | 786.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 780.20 | 784.07 | 785.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 782.30 | 780.92 | 783.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 782.30 | 780.92 | 783.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 782.30 | 780.92 | 783.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 781.55 | 780.92 | 783.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 790.65 | 782.64 | 783.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 790.65 | 782.64 | 783.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 788.45 | 783.80 | 783.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:15:00 | 788.60 | 783.80 | 783.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 788.30 | 784.70 | 784.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 793.90 | 788.46 | 786.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 15:15:00 | 819.90 | 821.07 | 814.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:15:00 | 825.00 | 821.07 | 814.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 823.40 | 824.17 | 820.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 827.35 | 824.72 | 821.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 826.75 | 824.38 | 821.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 821.65 | 823.69 | 822.52 | SL hit (close<ema400) qty=1.00 sl=822.52 alert=retest1 |

### Cycle 30 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 811.50 | 820.11 | 821.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 810.65 | 818.22 | 820.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 815.30 | 814.57 | 817.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 11:15:00 | 815.30 | 814.57 | 817.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 815.30 | 814.57 | 817.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:45:00 | 816.05 | 814.57 | 817.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 806.25 | 810.05 | 813.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 809.00 | 810.05 | 813.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 813.25 | 810.78 | 813.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 813.60 | 810.78 | 813.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 817.25 | 812.07 | 813.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 817.25 | 812.07 | 813.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 814.00 | 812.46 | 813.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 812.35 | 812.46 | 813.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 817.85 | 813.54 | 813.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 819.50 | 813.54 | 813.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 820.80 | 814.99 | 814.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 12:15:00 | 830.35 | 819.09 | 816.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 823.85 | 827.22 | 823.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 823.85 | 827.22 | 823.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 823.85 | 827.22 | 823.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 823.85 | 827.22 | 823.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 819.90 | 825.76 | 823.23 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 813.60 | 821.57 | 821.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 811.55 | 818.36 | 820.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 800.70 | 799.00 | 807.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 800.70 | 799.00 | 807.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 784.35 | 774.95 | 777.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 784.35 | 774.95 | 777.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 779.95 | 775.95 | 777.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 793.95 | 775.95 | 777.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 798.40 | 780.44 | 779.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 803.80 | 791.70 | 785.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 804.45 | 805.04 | 798.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 801.75 | 805.04 | 798.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 797.50 | 802.78 | 798.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 797.50 | 802.78 | 798.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 798.60 | 801.95 | 798.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 799.60 | 801.95 | 798.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 793.45 | 798.71 | 797.78 | SL hit (close<static) qty=1.00 sl=795.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 794.30 | 797.31 | 797.33 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 799.10 | 797.53 | 797.42 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 794.45 | 797.27 | 797.50 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 807.75 | 799.37 | 798.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 819.85 | 805.21 | 801.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 816.10 | 816.28 | 810.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 816.10 | 816.28 | 810.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 819.75 | 821.43 | 819.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 820.00 | 821.43 | 819.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 825.35 | 822.22 | 819.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 818.70 | 822.22 | 819.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 818.60 | 821.91 | 820.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 818.60 | 821.91 | 820.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 817.50 | 821.03 | 820.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 821.05 | 821.03 | 820.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 820.00 | 820.47 | 820.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 13:45:00 | 820.20 | 820.23 | 820.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 818.85 | 819.95 | 819.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 818.85 | 819.95 | 819.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 15:15:00 | 818.00 | 819.56 | 819.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 820.60 | 819.77 | 819.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 820.60 | 819.77 | 819.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 820.60 | 819.77 | 819.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 821.00 | 819.77 | 819.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 819.80 | 819.78 | 819.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 820.20 | 819.78 | 819.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 819.00 | 819.62 | 819.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:30:00 | 815.30 | 819.38 | 819.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 815.00 | 818.50 | 819.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 818.20 | 809.94 | 809.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 818.20 | 809.94 | 809.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 820.25 | 814.72 | 812.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 15:15:00 | 819.00 | 820.36 | 816.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 09:15:00 | 819.00 | 820.36 | 816.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 813.85 | 819.06 | 816.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 813.85 | 819.06 | 816.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 814.95 | 818.24 | 816.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 814.50 | 818.24 | 816.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 814.30 | 817.45 | 816.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 814.20 | 817.45 | 816.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 813.65 | 816.46 | 816.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 813.65 | 816.46 | 816.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 814.25 | 816.02 | 815.90 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 813.10 | 815.43 | 815.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 823.10 | 816.97 | 816.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 829.35 | 819.44 | 817.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 09:15:00 | 829.70 | 829.73 | 824.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:30:00 | 830.75 | 829.73 | 824.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 822.30 | 828.18 | 826.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 822.30 | 828.18 | 826.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 818.40 | 826.23 | 825.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 818.40 | 826.23 | 825.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 820.60 | 825.10 | 825.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 814.15 | 822.01 | 823.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 804.00 | 802.32 | 808.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 804.00 | 802.32 | 808.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 794.55 | 801.00 | 805.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 794.20 | 798.94 | 803.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 792.20 | 797.21 | 801.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 793.70 | 796.48 | 799.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 793.80 | 795.76 | 798.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 790.20 | 790.85 | 794.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 790.20 | 790.85 | 794.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 789.50 | 785.78 | 787.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 789.65 | 785.78 | 787.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 790.60 | 786.75 | 788.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 790.60 | 786.75 | 788.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 790.50 | 787.50 | 788.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 794.10 | 789.06 | 788.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 794.10 | 789.06 | 788.96 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 787.65 | 788.78 | 788.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 785.20 | 787.94 | 788.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 785.40 | 785.35 | 786.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 785.40 | 785.35 | 786.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 785.40 | 785.35 | 786.80 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 798.15 | 788.77 | 788.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 801.00 | 791.21 | 789.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 794.50 | 797.09 | 794.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 794.50 | 797.09 | 794.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 794.50 | 797.09 | 794.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 794.50 | 797.09 | 794.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 795.00 | 796.67 | 794.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 789.85 | 796.67 | 794.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 786.60 | 794.66 | 793.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 786.60 | 794.66 | 793.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 780.20 | 791.77 | 792.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 777.65 | 783.77 | 787.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 782.95 | 781.63 | 784.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 782.95 | 781.63 | 784.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 782.95 | 781.63 | 784.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 782.95 | 781.63 | 784.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 784.35 | 781.84 | 784.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 784.35 | 781.84 | 784.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 780.00 | 781.47 | 783.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 778.35 | 781.25 | 783.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 792.50 | 783.64 | 783.85 | SL hit (close>static) qty=1.00 sl=785.85 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 794.00 | 785.71 | 784.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 799.00 | 788.37 | 786.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 807.50 | 814.23 | 808.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 807.50 | 814.23 | 808.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 807.50 | 814.23 | 808.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 807.50 | 814.23 | 808.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 804.00 | 812.18 | 808.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 810.25 | 812.18 | 808.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 796.00 | 806.74 | 807.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 796.00 | 806.74 | 807.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 792.10 | 803.81 | 805.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 782.10 | 777.88 | 783.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 782.10 | 777.88 | 783.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 780.75 | 778.46 | 783.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 780.75 | 778.46 | 783.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 775.25 | 777.90 | 782.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 772.85 | 777.90 | 782.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:00:00 | 772.20 | 776.76 | 781.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 734.21 | 746.46 | 758.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 733.59 | 746.46 | 758.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 739.50 | 732.73 | 744.16 | SL hit (close>ema200) qty=0.50 sl=732.73 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 744.55 | 730.41 | 729.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 748.00 | 733.93 | 731.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 759.35 | 761.22 | 751.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:45:00 | 758.85 | 761.22 | 751.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 818.50 | 820.69 | 816.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 819.80 | 820.69 | 816.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 817.00 | 819.95 | 816.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 827.40 | 819.95 | 816.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 837.30 | 842.26 | 842.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 837.30 | 842.26 | 842.79 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 846.95 | 842.93 | 842.86 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 841.90 | 842.73 | 842.78 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 12:15:00 | 844.75 | 843.19 | 842.98 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 837.55 | 842.02 | 842.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 832.85 | 839.47 | 841.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 816.80 | 813.99 | 820.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 816.80 | 813.99 | 820.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 816.80 | 813.99 | 820.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 809.40 | 813.07 | 818.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 810.00 | 812.84 | 817.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 808.75 | 812.02 | 817.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 810.00 | 811.94 | 816.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 818.20 | 812.88 | 816.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 818.00 | 812.88 | 816.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 827.15 | 815.73 | 817.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 827.15 | 815.73 | 817.21 | SL hit (close>static) qty=1.00 sl=823.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 832.00 | 818.99 | 818.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 837.80 | 822.75 | 820.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 828.15 | 828.21 | 824.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 12:00:00 | 828.15 | 828.21 | 824.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 823.70 | 826.95 | 824.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 824.10 | 826.95 | 824.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 822.00 | 825.96 | 824.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 822.00 | 825.96 | 824.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 819.65 | 824.70 | 824.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 818.25 | 824.70 | 824.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 818.80 | 823.52 | 823.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 811.00 | 818.19 | 820.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 792.85 | 790.43 | 799.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 792.85 | 790.43 | 799.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 792.85 | 790.43 | 799.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 797.80 | 790.43 | 799.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 791.85 | 756.59 | 759.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 798.00 | 756.59 | 759.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 783.80 | 762.03 | 761.36 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 760.65 | 767.35 | 768.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 756.60 | 765.20 | 767.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 772.30 | 753.82 | 756.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 772.30 | 753.82 | 756.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 772.30 | 753.82 | 756.99 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 780.90 | 759.23 | 759.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 783.90 | 764.17 | 761.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 782.60 | 782.78 | 775.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 782.60 | 782.78 | 775.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 776.75 | 780.94 | 776.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 776.75 | 780.94 | 776.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 782.85 | 781.32 | 777.34 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 766.35 | 775.72 | 776.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 764.45 | 773.47 | 774.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 771.85 | 765.41 | 769.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 771.85 | 765.41 | 769.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 771.85 | 765.41 | 769.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 771.85 | 765.41 | 769.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 770.80 | 766.49 | 769.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 770.80 | 766.49 | 769.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 773.00 | 767.79 | 769.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 773.70 | 767.79 | 769.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 773.95 | 769.14 | 770.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 774.10 | 769.14 | 770.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 771.55 | 769.62 | 770.17 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 784.65 | 773.02 | 771.65 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 768.00 | 774.97 | 775.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 766.00 | 770.54 | 772.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 761.80 | 757.61 | 762.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 10:15:00 | 761.80 | 757.61 | 762.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 761.80 | 757.61 | 762.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 761.80 | 757.61 | 762.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 772.00 | 760.49 | 763.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 772.00 | 760.49 | 763.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 763.20 | 761.03 | 763.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 781.35 | 761.03 | 763.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 764.10 | 761.64 | 763.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 764.10 | 761.64 | 763.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 763.20 | 761.95 | 763.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 762.15 | 761.95 | 763.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 762.55 | 762.07 | 763.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 762.15 | 762.07 | 763.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 764.00 | 762.46 | 763.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 764.00 | 762.46 | 763.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 760.85 | 762.14 | 763.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 755.50 | 759.91 | 761.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 757.35 | 758.63 | 760.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 756.55 | 758.63 | 760.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 766.65 | 760.66 | 761.36 | SL hit (close>static) qty=1.00 sl=764.65 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 793.40 | 767.21 | 764.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 795.55 | 777.26 | 769.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 797.00 | 797.64 | 787.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 803.25 | 797.86 | 788.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:15:00 | 807.95 | 797.86 | 788.21 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 787.40 | 800.29 | 795.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 787.40 | 800.29 | 795.02 | SL hit (close<ema400) qty=1.00 sl=795.02 alert=retest1 |

### Cycle 64 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 778.10 | 789.71 | 790.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 771.90 | 786.15 | 789.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 685.20 | 684.33 | 695.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 715.30 | 684.33 | 695.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 715.95 | 690.65 | 697.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 708.10 | 690.65 | 697.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 705.15 | 693.55 | 698.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 702.85 | 696.16 | 698.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 710.20 | 701.32 | 700.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 710.20 | 701.32 | 700.95 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 663.40 | 694.05 | 697.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 651.10 | 685.46 | 693.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 678.00 | 664.59 | 676.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 678.00 | 664.59 | 676.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 678.00 | 664.59 | 676.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 678.00 | 664.59 | 676.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 672.50 | 666.17 | 676.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:30:00 | 677.45 | 666.17 | 676.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 673.75 | 667.69 | 676.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:30:00 | 679.90 | 667.69 | 676.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 673.90 | 668.93 | 676.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 673.90 | 668.93 | 676.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 663.55 | 668.41 | 674.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 674.10 | 668.41 | 674.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 665.70 | 667.42 | 673.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 660.95 | 665.98 | 671.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 627.90 | 639.64 | 648.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 632.15 | 622.57 | 630.75 | SL hit (close>ema200) qty=0.50 sl=622.57 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 644.75 | 634.78 | 633.75 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 621.70 | 634.45 | 634.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 620.10 | 631.58 | 633.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 618.30 | 618.08 | 623.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 14:00:00 | 618.30 | 618.08 | 623.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 611.60 | 591.99 | 601.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 616.05 | 591.99 | 601.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 611.15 | 595.82 | 602.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 611.85 | 595.82 | 602.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 618.50 | 608.34 | 607.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 638.70 | 616.28 | 611.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 641.95 | 643.37 | 630.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 641.95 | 643.37 | 630.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 623.25 | 637.99 | 634.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 621.65 | 637.99 | 634.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 615.30 | 629.64 | 630.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 608.05 | 621.56 | 626.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 638.85 | 623.17 | 626.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 638.85 | 623.17 | 626.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 638.85 | 623.17 | 626.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 640.30 | 623.17 | 626.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 636.70 | 625.88 | 627.19 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 645.80 | 629.86 | 628.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 654.00 | 634.69 | 631.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 620.00 | 634.17 | 632.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 620.00 | 634.17 | 632.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 620.00 | 634.17 | 632.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 620.00 | 634.17 | 632.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 624.45 | 632.22 | 631.67 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 624.75 | 630.73 | 631.04 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 636.50 | 631.57 | 631.30 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 628.20 | 630.64 | 630.91 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 636.95 | 631.66 | 631.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 641.45 | 633.62 | 632.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 635.00 | 637.25 | 634.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 635.00 | 637.25 | 634.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 635.00 | 637.25 | 634.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 633.85 | 637.25 | 634.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 638.25 | 637.45 | 635.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:15:00 | 645.00 | 638.94 | 636.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:45:00 | 645.75 | 639.74 | 637.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 661.95 | 640.38 | 637.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 09:15:00 | 709.50 | 686.83 | 673.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 716.50 | 722.70 | 722.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 714.25 | 719.70 | 721.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 725.80 | 717.22 | 719.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 725.80 | 717.22 | 719.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 725.80 | 717.22 | 719.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 725.80 | 717.22 | 719.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 721.00 | 717.98 | 719.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:30:00 | 717.50 | 718.19 | 719.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 713.65 | 717.85 | 718.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:15:00 | 681.62 | 691.69 | 699.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 677.97 | 684.75 | 694.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 15:15:00 | 664.00 | 663.63 | 671.61 | SL hit (close>ema200) qty=0.50 sl=663.63 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 675.90 | 668.43 | 667.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 686.55 | 672.05 | 669.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 686.25 | 688.32 | 682.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 686.65 | 688.32 | 682.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 683.70 | 687.21 | 683.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 680.25 | 687.21 | 683.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 684.10 | 686.59 | 683.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:15:00 | 683.55 | 686.59 | 683.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 687.95 | 686.86 | 683.63 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-26 09:15:00 | 713.80 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-27 10:45:00 | 707.10 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-05-27 12:00:00 | 706.75 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-05-27 13:30:00 | 707.25 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-05-28 09:15:00 | 711.95 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-29 09:45:00 | 706.05 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-05-29 10:15:00 | 706.35 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-05-29 10:45:00 | 707.10 | 2025-05-29 13:15:00 | 704.70 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-06 14:15:00 | 701.30 | 2025-06-09 09:15:00 | 713.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-23 14:45:00 | 705.20 | 2025-06-27 15:15:00 | 701.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-08 09:15:00 | 714.50 | 2025-07-08 10:15:00 | 705.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-30 11:00:00 | 738.05 | 2025-07-30 14:15:00 | 744.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-30 11:30:00 | 737.65 | 2025-07-30 14:15:00 | 744.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-08-04 09:15:00 | 759.15 | 2025-08-05 09:15:00 | 749.30 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-04 11:30:00 | 757.50 | 2025-08-05 09:15:00 | 749.30 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-06 10:45:00 | 742.65 | 2025-08-11 11:15:00 | 763.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-08-06 13:00:00 | 742.95 | 2025-08-11 11:15:00 | 763.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-08-06 13:45:00 | 742.45 | 2025-08-11 11:15:00 | 763.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-08-08 15:00:00 | 741.50 | 2025-08-11 11:15:00 | 763.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-08-25 10:45:00 | 800.85 | 2025-08-25 11:15:00 | 809.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-29 13:00:00 | 791.50 | 2025-09-01 09:15:00 | 803.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-01 14:00:00 | 792.85 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-09-01 15:00:00 | 791.00 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-09-02 09:15:00 | 793.00 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-09-04 10:30:00 | 778.50 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-04 11:30:00 | 779.35 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-04 14:45:00 | 779.15 | 2025-09-08 09:15:00 | 790.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest1 | 2025-09-16 09:15:00 | 825.00 | 2025-09-18 12:15:00 | 821.65 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-17 13:15:00 | 827.35 | 2025-09-18 14:15:00 | 818.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-17 13:45:00 | 826.75 | 2025-09-18 14:15:00 | 818.80 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-08 13:15:00 | 799.60 | 2025-10-08 15:15:00 | 793.45 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-20 09:15:00 | 821.05 | 2025-10-20 14:15:00 | 818.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-10-20 12:15:00 | 820.00 | 2025-10-20 14:15:00 | 818.85 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-10-20 13:45:00 | 820.20 | 2025-10-20 14:15:00 | 818.85 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-23 10:30:00 | 815.30 | 2025-10-28 12:15:00 | 818.20 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-23 12:00:00 | 815.00 | 2025-10-28 12:15:00 | 818.20 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-10 12:15:00 | 794.20 | 2025-11-17 13:15:00 | 794.10 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-11-10 15:15:00 | 792.20 | 2025-11-17 13:15:00 | 794.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-11 15:00:00 | 793.70 | 2025-11-17 13:15:00 | 794.10 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-11-12 09:30:00 | 793.80 | 2025-11-17 13:15:00 | 794.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-25 12:15:00 | 778.35 | 2025-11-26 09:15:00 | 792.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-01 09:15:00 | 810.25 | 2025-12-01 14:15:00 | 796.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-05 10:15:00 | 772.85 | 2025-12-09 09:15:00 | 734.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:00:00 | 772.20 | 2025-12-09 09:15:00 | 733.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 10:15:00 | 772.85 | 2025-12-10 09:15:00 | 739.50 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2025-12-05 11:00:00 | 772.20 | 2025-12-10 09:15:00 | 739.50 | STOP_HIT | 0.50 | 4.23% |
| BUY | retest2 | 2025-12-30 09:15:00 | 827.40 | 2026-01-06 11:15:00 | 837.30 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2026-01-13 12:00:00 | 809.40 | 2026-01-14 10:15:00 | 827.15 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-13 12:45:00 | 810.00 | 2026-01-14 10:15:00 | 827.15 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-13 14:00:00 | 808.75 | 2026-01-14 10:15:00 | 827.15 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-13 15:15:00 | 810.00 | 2026-01-14 10:15:00 | 827.15 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-02-17 15:00:00 | 755.50 | 2026-02-18 11:15:00 | 766.65 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-02-18 09:45:00 | 757.35 | 2026-02-18 11:15:00 | 766.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-18 10:15:00 | 756.55 | 2026-02-18 11:15:00 | 766.65 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest1 | 2026-02-20 09:45:00 | 803.25 | 2026-02-23 09:15:00 | 787.40 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2026-02-20 10:15:00 | 807.95 | 2026-02-23 09:15:00 | 787.40 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-03-06 11:30:00 | 702.85 | 2026-03-06 13:15:00 | 710.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-11 10:30:00 | 660.95 | 2026-03-13 13:15:00 | 627.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 660.95 | 2026-03-17 09:15:00 | 632.15 | STOP_HIT | 0.50 | 4.36% |
| BUY | retest2 | 2026-04-07 13:15:00 | 645.00 | 2026-04-10 09:15:00 | 709.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:45:00 | 645.75 | 2026-04-10 09:15:00 | 710.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 661.95 | 2026-04-16 09:15:00 | 728.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 12:30:00 | 717.50 | 2026-04-28 12:15:00 | 681.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 713.65 | 2026-04-28 15:15:00 | 677.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 12:30:00 | 717.50 | 2026-04-30 15:15:00 | 664.00 | STOP_HIT | 0.50 | 7.46% |
| SELL | retest2 | 2026-04-24 09:15:00 | 713.65 | 2026-04-30 15:15:00 | 664.00 | STOP_HIT | 0.50 | 6.96% |
