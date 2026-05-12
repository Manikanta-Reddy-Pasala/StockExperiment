# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 630.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 54 |
| ALERT2 | 55 |
| ALERT2_SKIP | 23 |
| ALERT3 | 146 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 18 / 37
- **Target hits / Stop hits / Partials:** 0 / 49 / 6
- **Avg / median % per leg:** 0.56% / -0.60%
- **Sum % (uncompounded):** 31.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 14 | 0 | -0.23% | -3.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 14 | 0 | -0.23% | -3.2% |
| SELL (all) | 41 | 15 | 36.6% | 0 | 35 | 6 | 0.84% | 34.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 15 | 36.6% | 0 | 35 | 6 | 0.84% | 34.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 18 | 32.7% | 0 | 49 | 6 | 0.56% | 31.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 757.00 | 766.24 | 766.88 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 770.15 | 766.97 | 766.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 773.05 | 768.18 | 767.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 772.10 | 775.41 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 772.10 | 775.41 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 771.70 | 774.67 | 771.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 776.80 | 774.67 | 771.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 776.25 | 774.98 | 772.37 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 767.20 | 771.46 | 772.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 757.75 | 767.78 | 770.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 763.35 | 762.41 | 765.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 759.15 | 764.08 | 765.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 771.75 | 761.75 | 762.39 | SL hit (close>static) qty=1.00 sl=765.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 772.20 | 763.84 | 763.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 774.85 | 767.57 | 765.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 783.00 | 787.70 | 782.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 783.40 | 787.70 | 782.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 785.95 | 787.35 | 782.60 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 775.05 | 779.96 | 780.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 764.15 | 776.80 | 778.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 751.95 | 748.44 | 756.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 751.95 | 748.44 | 756.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 732.75 | 734.56 | 740.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 727.70 | 733.48 | 739.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 721.85 | 706.30 | 704.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 721.85 | 706.30 | 704.62 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 703.00 | 708.83 | 709.41 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 710.55 | 708.81 | 708.65 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 707.35 | 708.52 | 708.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 703.25 | 706.10 | 707.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 704.85 | 702.33 | 703.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 704.85 | 702.33 | 703.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 706.90 | 703.24 | 704.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 707.00 | 703.24 | 704.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 704.50 | 704.12 | 704.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 703.00 | 704.12 | 704.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 711.30 | 705.50 | 704.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 711.30 | 705.50 | 704.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 714.80 | 710.17 | 708.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 709.95 | 711.67 | 709.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 709.95 | 711.67 | 709.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 708.20 | 710.98 | 709.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 705.00 | 710.98 | 709.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 709.00 | 710.58 | 709.25 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 706.00 | 708.20 | 708.37 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 710.35 | 708.63 | 708.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 715.80 | 710.07 | 709.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 730.10 | 732.91 | 727.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 730.45 | 732.91 | 727.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 727.60 | 731.51 | 727.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 727.60 | 731.51 | 727.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 725.30 | 730.27 | 727.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 725.30 | 730.27 | 727.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 719.50 | 725.75 | 726.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 715.70 | 721.95 | 724.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 716.15 | 716.15 | 719.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 716.15 | 716.15 | 719.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 715.00 | 715.95 | 718.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 711.40 | 715.13 | 716.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 711.35 | 714.65 | 716.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 718.70 | 716.02 | 715.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 718.70 | 716.02 | 715.87 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 709.80 | 714.85 | 715.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 708.65 | 713.31 | 714.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 712.70 | 711.73 | 713.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 712.75 | 711.73 | 713.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 709.90 | 711.37 | 713.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 707.05 | 710.10 | 712.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 707.60 | 704.78 | 706.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:00:00 | 707.10 | 705.64 | 706.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 711.00 | 707.69 | 707.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 711.00 | 707.69 | 707.56 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 703.25 | 707.61 | 707.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 695.20 | 703.54 | 705.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 694.70 | 693.57 | 697.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:30:00 | 693.60 | 693.57 | 697.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 690.05 | 692.90 | 696.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 689.00 | 691.60 | 694.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 687.40 | 690.42 | 693.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 654.55 | 664.95 | 674.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 653.03 | 664.95 | 674.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 667.20 | 665.40 | 673.60 | SL hit (close>ema200) qty=0.50 sl=665.40 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 669.55 | 664.15 | 663.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 674.65 | 666.95 | 665.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 667.75 | 668.14 | 666.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 667.75 | 668.14 | 666.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 670.05 | 668.52 | 666.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 668.00 | 668.52 | 666.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 685.10 | 687.71 | 685.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 684.40 | 687.71 | 685.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 683.90 | 686.95 | 685.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:15:00 | 684.90 | 686.95 | 685.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 685.00 | 686.56 | 685.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 687.25 | 686.43 | 685.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 688.15 | 685.48 | 685.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:30:00 | 687.00 | 687.62 | 686.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 687.20 | 687.24 | 686.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 685.00 | 686.79 | 686.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 680.00 | 686.79 | 686.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 681.50 | 685.74 | 685.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 681.50 | 685.74 | 685.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 671.85 | 679.18 | 682.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 665.50 | 663.28 | 668.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 668.35 | 663.28 | 668.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 671.65 | 665.39 | 668.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 671.65 | 665.39 | 668.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 672.70 | 666.85 | 668.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 672.70 | 666.85 | 668.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 676.60 | 670.06 | 670.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 678.40 | 671.73 | 670.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 681.35 | 681.81 | 678.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 681.35 | 681.81 | 678.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 680.55 | 681.50 | 678.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 678.90 | 681.50 | 678.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 682.50 | 682.80 | 681.03 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 670.75 | 678.42 | 679.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 669.65 | 676.67 | 678.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 674.00 | 673.44 | 675.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 676.75 | 673.44 | 675.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 680.95 | 674.94 | 676.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 679.80 | 674.94 | 676.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 681.00 | 676.15 | 676.75 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 681.60 | 677.24 | 677.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 686.75 | 679.89 | 678.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 680.25 | 680.46 | 679.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 680.25 | 680.46 | 679.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 680.40 | 680.44 | 679.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 680.30 | 680.44 | 679.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 679.60 | 680.28 | 679.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 679.50 | 680.28 | 679.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 680.95 | 680.41 | 679.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 688.40 | 680.41 | 679.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 705.50 | 713.77 | 714.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 705.50 | 713.77 | 714.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 702.85 | 709.00 | 712.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 677.30 | 672.68 | 677.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 678.85 | 672.68 | 677.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 676.50 | 673.45 | 677.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 673.55 | 675.16 | 676.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 679.00 | 673.51 | 674.87 | SL hit (close>static) qty=1.00 sl=677.60 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 684.45 | 677.04 | 676.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 691.30 | 679.89 | 677.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 711.95 | 712.72 | 707.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 11:15:00 | 708.00 | 711.26 | 707.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 708.00 | 711.26 | 707.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 708.40 | 711.26 | 707.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 709.40 | 710.89 | 707.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 711.45 | 710.89 | 707.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 712.05 | 709.61 | 707.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 706.15 | 713.08 | 712.53 | SL hit (close<static) qty=1.00 sl=706.50 alert=retest2 |

### Cycle 25 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 701.50 | 710.76 | 711.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 695.35 | 702.67 | 706.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 695.70 | 693.69 | 699.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 695.70 | 693.69 | 699.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 697.65 | 695.09 | 697.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 697.25 | 695.09 | 697.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 691.50 | 694.37 | 697.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 689.20 | 693.14 | 695.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:45:00 | 690.75 | 692.82 | 695.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:45:00 | 689.55 | 692.22 | 694.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 698.70 | 688.31 | 687.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 698.70 | 688.31 | 687.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 700.00 | 695.66 | 693.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 695.00 | 695.81 | 694.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 12:00:00 | 695.00 | 695.81 | 694.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 694.80 | 695.53 | 694.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 694.80 | 695.53 | 694.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 694.85 | 695.39 | 694.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 693.80 | 695.39 | 694.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 699.45 | 699.54 | 698.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 698.30 | 699.54 | 698.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 697.50 | 698.90 | 698.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 697.50 | 698.90 | 698.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 697.20 | 698.56 | 698.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:15:00 | 697.35 | 698.56 | 698.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 696.20 | 698.09 | 698.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 696.20 | 698.09 | 698.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 695.50 | 697.57 | 697.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 691.30 | 696.32 | 697.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 696.40 | 692.88 | 694.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 696.40 | 692.88 | 694.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 697.40 | 693.78 | 694.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 695.00 | 693.78 | 694.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 677.50 | 674.89 | 678.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 677.35 | 674.89 | 678.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 679.70 | 675.85 | 678.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 681.35 | 675.85 | 678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 676.90 | 676.06 | 678.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 676.25 | 676.10 | 678.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 676.35 | 677.42 | 678.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 676.30 | 677.20 | 678.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 682.00 | 678.52 | 678.57 | SL hit (close>static) qty=1.00 sl=680.65 alert=retest2 |

### Cycle 28 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 684.95 | 679.81 | 679.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 696.65 | 683.83 | 681.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 693.25 | 694.93 | 690.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 692.80 | 694.93 | 690.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 690.35 | 694.01 | 690.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 690.35 | 694.01 | 690.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 686.20 | 692.45 | 690.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 686.20 | 692.45 | 690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 686.00 | 691.16 | 689.97 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 681.05 | 689.14 | 689.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 678.35 | 685.14 | 687.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 12:15:00 | 683.20 | 682.32 | 684.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 12:45:00 | 683.55 | 682.32 | 684.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 677.35 | 677.74 | 680.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 680.30 | 677.74 | 680.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 678.15 | 677.82 | 680.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 679.50 | 677.82 | 680.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 683.90 | 679.04 | 680.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 683.90 | 679.04 | 680.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 683.00 | 679.83 | 680.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 682.30 | 679.83 | 680.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:15:00 | 682.40 | 680.43 | 680.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 15:15:00 | 682.75 | 681.24 | 681.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 682.75 | 681.24 | 681.13 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 673.85 | 679.90 | 680.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 670.75 | 675.11 | 677.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 676.75 | 674.66 | 677.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 678.30 | 674.66 | 677.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 678.90 | 675.51 | 677.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 678.90 | 675.51 | 677.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 679.00 | 676.21 | 677.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 681.35 | 676.21 | 677.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 677.00 | 677.29 | 677.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 677.00 | 677.29 | 677.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 676.85 | 677.20 | 677.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 673.75 | 677.20 | 677.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 675.15 | 673.17 | 674.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 679.95 | 675.81 | 675.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 679.95 | 675.81 | 675.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 681.60 | 677.75 | 676.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 679.35 | 679.95 | 678.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 679.35 | 679.95 | 678.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 677.00 | 679.36 | 677.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 677.00 | 679.36 | 677.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 674.60 | 678.41 | 677.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 674.60 | 678.41 | 677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 677.50 | 677.68 | 677.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 679.15 | 677.68 | 677.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 679.00 | 678.23 | 677.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 680.00 | 679.19 | 678.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 11:15:00 | 675.05 | 678.27 | 678.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 675.05 | 678.27 | 678.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 672.50 | 675.87 | 676.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 677.65 | 674.91 | 676.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 677.45 | 674.91 | 676.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 675.55 | 675.04 | 676.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 674.40 | 675.08 | 676.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 675.10 | 675.63 | 676.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 674.45 | 675.40 | 676.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 674.70 | 675.42 | 675.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 666.50 | 673.52 | 675.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 664.90 | 669.32 | 672.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 663.00 | 667.24 | 670.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 641.35 | 654.72 | 661.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.68 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.73 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 15:15:00 | 640.97 | 646.74 | 651.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 647.50 | 646.89 | 651.46 | SL hit (close>ema200) qty=0.50 sl=646.89 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 655.60 | 652.47 | 652.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 659.70 | 654.89 | 653.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 656.60 | 657.20 | 655.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 656.60 | 657.20 | 655.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 657.50 | 657.26 | 655.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 656.55 | 657.26 | 655.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 651.05 | 656.02 | 655.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 651.05 | 656.02 | 655.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 651.20 | 655.05 | 655.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 650.00 | 653.35 | 654.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 644.75 | 643.95 | 647.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 645.35 | 643.95 | 647.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 644.80 | 643.41 | 645.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 646.85 | 643.41 | 645.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 646.20 | 643.97 | 645.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 648.10 | 643.97 | 645.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 647.85 | 644.74 | 645.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 647.25 | 644.74 | 645.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 646.90 | 645.18 | 646.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 648.00 | 645.18 | 646.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 652.65 | 646.67 | 646.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 653.80 | 648.10 | 647.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 661.30 | 662.70 | 659.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 664.15 | 662.70 | 659.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 659.75 | 661.68 | 660.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 658.80 | 661.68 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 658.00 | 660.94 | 659.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 657.40 | 660.94 | 659.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 659.15 | 660.01 | 659.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 659.25 | 660.01 | 659.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 658.80 | 659.77 | 659.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 658.85 | 659.77 | 659.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 658.40 | 659.49 | 659.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 657.20 | 659.04 | 659.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 643.00 | 642.69 | 646.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 643.00 | 642.69 | 646.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 641.80 | 642.74 | 645.28 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 652.45 | 645.21 | 645.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 653.20 | 646.81 | 645.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 651.85 | 653.61 | 651.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 651.85 | 653.61 | 651.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 651.25 | 653.14 | 651.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 649.85 | 653.14 | 651.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 650.50 | 652.61 | 650.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 649.55 | 652.61 | 650.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 650.50 | 652.19 | 650.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 649.05 | 652.19 | 650.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 648.40 | 651.43 | 650.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:15:00 | 647.85 | 651.43 | 650.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 647.55 | 650.12 | 650.23 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 682.00 | 656.24 | 652.96 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 655.70 | 663.06 | 663.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 654.80 | 661.41 | 663.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 652.25 | 651.45 | 655.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 654.30 | 651.45 | 655.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 650.10 | 651.25 | 654.66 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 659.85 | 655.44 | 655.39 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 651.30 | 654.61 | 655.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 11:15:00 | 648.25 | 652.84 | 654.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 655.45 | 650.76 | 652.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 656.50 | 650.76 | 652.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 653.30 | 651.27 | 652.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 652.80 | 651.27 | 652.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 654.00 | 651.82 | 652.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:30:00 | 652.65 | 651.82 | 652.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 650.70 | 651.59 | 652.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 655.00 | 651.59 | 652.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 649.90 | 651.25 | 652.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 651.35 | 651.25 | 652.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 650.35 | 651.07 | 651.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 651.90 | 651.07 | 651.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 651.45 | 651.15 | 651.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 669.10 | 651.15 | 651.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 676.35 | 656.19 | 654.15 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 650.80 | 657.13 | 657.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 647.10 | 655.13 | 656.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 659.00 | 647.06 | 649.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 660.40 | 647.06 | 649.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 658.80 | 649.41 | 650.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 658.80 | 649.41 | 650.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 656.40 | 650.80 | 650.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 659.75 | 653.51 | 652.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 654.10 | 654.25 | 652.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 654.10 | 654.25 | 652.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 659.90 | 655.38 | 653.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 650.10 | 655.38 | 653.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 655.10 | 655.72 | 654.17 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 648.05 | 652.66 | 653.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 645.30 | 651.19 | 652.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 651.45 | 651.24 | 652.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 651.45 | 651.24 | 652.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 652.70 | 651.53 | 652.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 654.90 | 651.53 | 652.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 651.30 | 651.49 | 652.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 650.10 | 651.49 | 652.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 657.60 | 653.10 | 652.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 657.60 | 653.10 | 652.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 661.40 | 655.37 | 653.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 650.15 | 655.07 | 654.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 650.15 | 655.07 | 654.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 648.25 | 653.71 | 653.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 648.65 | 653.71 | 653.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 647.70 | 652.51 | 653.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 646.50 | 649.69 | 651.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 647.55 | 647.40 | 649.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:45:00 | 646.75 | 647.40 | 649.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 651.05 | 648.13 | 649.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 651.05 | 648.13 | 649.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 647.55 | 648.01 | 649.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 647.55 | 648.01 | 649.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 646.40 | 641.24 | 643.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 646.40 | 641.24 | 643.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 646.85 | 642.37 | 643.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 657.50 | 642.37 | 643.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 657.60 | 647.39 | 646.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 660.55 | 650.02 | 647.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 649.70 | 654.97 | 651.42 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 646.35 | 649.57 | 649.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 643.10 | 648.27 | 649.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 629.50 | 622.38 | 626.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:45:00 | 630.75 | 622.38 | 626.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 633.15 | 624.53 | 627.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 634.00 | 624.53 | 627.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 632.10 | 629.02 | 628.99 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 625.00 | 628.58 | 628.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 618.30 | 625.51 | 627.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 597.95 | 597.53 | 603.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 601.10 | 597.53 | 603.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 608.25 | 599.68 | 603.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 608.30 | 599.68 | 603.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 605.90 | 600.92 | 603.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 604.25 | 604.81 | 605.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 603.10 | 604.78 | 605.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 603.00 | 603.87 | 604.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 603.35 | 603.60 | 604.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 603.35 | 603.55 | 604.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 607.80 | 603.55 | 604.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 608.85 | 604.61 | 604.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 608.85 | 604.61 | 604.46 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 599.50 | 604.31 | 604.62 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 607.30 | 604.91 | 604.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 608.35 | 606.07 | 605.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 606.00 | 606.66 | 605.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 609.10 | 606.66 | 605.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 602.95 | 605.89 | 605.74 | SL hit (close<static) qty=1.00 sl=605.70 alert=retest2 |

### Cycle 57 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 601.65 | 605.04 | 605.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 15:15:00 | 598.35 | 602.19 | 603.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 581.00 | 580.75 | 587.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 581.00 | 580.75 | 587.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 586.40 | 580.40 | 584.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 585.35 | 580.40 | 584.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 583.80 | 581.08 | 584.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 584.20 | 581.08 | 584.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 582.60 | 580.35 | 583.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 582.60 | 580.35 | 583.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 584.00 | 581.08 | 583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 589.80 | 581.08 | 583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 584.20 | 581.70 | 583.30 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 587.00 | 584.35 | 584.18 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 578.20 | 583.49 | 583.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 575.20 | 581.83 | 583.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 579.75 | 579.56 | 581.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 579.55 | 579.56 | 581.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 582.50 | 580.15 | 581.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 573.10 | 580.15 | 581.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 577.60 | 576.25 | 576.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 578.10 | 576.84 | 576.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 578.10 | 576.84 | 576.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 578.55 | 577.18 | 576.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 576.50 | 577.04 | 576.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 576.50 | 577.04 | 576.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 578.00 | 577.23 | 576.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 561.75 | 577.23 | 576.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 562.55 | 574.30 | 575.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 556.35 | 562.20 | 565.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 529.85 | 529.84 | 537.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:45:00 | 530.50 | 529.84 | 537.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 546.00 | 531.18 | 534.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 546.00 | 531.18 | 534.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 558.55 | 536.66 | 536.48 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 528.05 | 538.46 | 539.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 525.70 | 535.91 | 538.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 533.65 | 530.78 | 534.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 532.85 | 530.78 | 534.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 537.00 | 532.03 | 534.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 537.00 | 532.03 | 534.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 537.45 | 533.11 | 534.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 537.30 | 533.11 | 534.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 538.65 | 534.22 | 535.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 537.90 | 534.22 | 535.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 539.50 | 536.63 | 536.30 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 529.75 | 534.95 | 535.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 522.10 | 531.31 | 533.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 532.35 | 526.26 | 529.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 532.35 | 526.26 | 529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 534.10 | 527.83 | 529.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 534.60 | 527.83 | 529.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 546.25 | 532.96 | 531.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 547.90 | 535.95 | 533.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 541.15 | 541.86 | 538.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 539.40 | 541.86 | 538.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 538.50 | 541.19 | 538.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 538.00 | 541.19 | 538.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 541.70 | 541.29 | 538.43 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 522.50 | 535.38 | 536.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 518.40 | 531.98 | 534.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 537.40 | 522.61 | 527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 537.40 | 522.61 | 527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 533.55 | 524.80 | 528.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 530.75 | 529.72 | 529.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 531.20 | 530.14 | 530.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 531.20 | 530.14 | 530.05 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 519.00 | 527.91 | 529.05 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 540.00 | 530.52 | 529.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 541.80 | 532.78 | 530.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 555.90 | 556.92 | 551.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 555.90 | 556.92 | 551.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 556.10 | 562.47 | 559.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 563.60 | 562.40 | 560.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 572.35 | 561.25 | 560.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 575.50 | 581.27 | 581.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 575.50 | 581.27 | 581.58 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 584.70 | 581.96 | 581.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 585.80 | 583.17 | 582.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 582.90 | 583.72 | 582.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 582.90 | 583.72 | 582.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 584.85 | 583.95 | 583.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 574.40 | 583.95 | 583.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 577.80 | 582.72 | 582.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 578.25 | 582.72 | 582.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 575.05 | 581.19 | 581.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 573.65 | 579.68 | 581.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 562.90 | 562.30 | 567.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 13:45:00 | 563.00 | 562.30 | 567.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 571.10 | 564.06 | 568.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 571.10 | 564.06 | 568.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 565.00 | 564.25 | 567.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 576.25 | 564.25 | 567.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 575.10 | 566.42 | 568.47 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 580.15 | 570.62 | 570.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 588.55 | 574.20 | 571.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 582.00 | 583.60 | 577.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:00:00 | 582.00 | 583.60 | 577.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 579.15 | 582.49 | 578.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 579.15 | 582.49 | 578.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 577.65 | 581.52 | 578.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 577.65 | 581.52 | 578.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 573.40 | 579.90 | 577.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 573.40 | 579.90 | 577.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 569.00 | 577.72 | 577.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 569.00 | 577.72 | 577.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 570.90 | 576.35 | 576.47 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 588.80 | 576.85 | 575.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 638.05 | 596.17 | 586.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 625.00 | 625.97 | 617.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 625.00 | 625.97 | 617.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 626.75 | 629.66 | 626.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 10:45:00 | 634.05 | 630.79 | 627.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 09:30:00 | 760.50 | 2025-05-28 14:15:00 | 757.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-06 09:15:00 | 759.15 | 2025-06-09 09:15:00 | 771.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-19 11:15:00 | 727.70 | 2025-06-27 09:15:00 | 721.85 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-07-09 14:15:00 | 703.00 | 2025-07-10 09:15:00 | 711.30 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-23 09:30:00 | 711.40 | 2025-07-24 12:15:00 | 718.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-23 11:15:00 | 711.35 | 2025-07-24 12:15:00 | 718.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-28 11:30:00 | 707.05 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-29 15:00:00 | 707.60 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-30 10:00:00 | 707.10 | 2025-07-30 12:15:00 | 711.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-05 12:45:00 | 689.00 | 2025-08-07 13:15:00 | 654.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 15:15:00 | 687.40 | 2025-08-07 13:15:00 | 653.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 12:45:00 | 689.00 | 2025-08-07 14:15:00 | 667.20 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-08-05 15:15:00 | 687.40 | 2025-08-07 14:15:00 | 667.20 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2025-08-22 13:30:00 | 687.25 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-25 09:15:00 | 688.15 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-08-25 13:30:00 | 687.00 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-25 14:45:00 | 687.20 | 2025-08-26 09:15:00 | 681.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-10 09:15:00 | 688.40 | 2025-09-22 11:15:00 | 705.50 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2025-09-30 13:00:00 | 673.55 | 2025-10-01 11:15:00 | 679.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-08 13:15:00 | 711.45 | 2025-10-13 09:15:00 | 706.15 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-09 09:15:00 | 712.05 | 2025-10-13 09:15:00 | 706.15 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-16 11:30:00 | 689.20 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-16 12:45:00 | 690.75 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-17 09:45:00 | 689.55 | 2025-10-23 10:15:00 | 698.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-11-10 13:00:00 | 676.25 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-11 10:15:00 | 676.35 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-11 11:00:00 | 676.30 | 2025-11-11 13:15:00 | 682.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-19 13:15:00 | 682.30 | 2025-11-19 15:15:00 | 682.75 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-19 14:15:00 | 682.40 | 2025-11-19 15:15:00 | 682.75 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-11-25 09:15:00 | 673.75 | 2025-11-26 12:15:00 | 679.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-26 09:30:00 | 675.15 | 2025-11-26 12:15:00 | 679.95 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-28 09:15:00 | 679.15 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-28 13:30:00 | 679.00 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-01 13:30:00 | 680.00 | 2025-12-02 11:15:00 | 675.05 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-04 11:45:00 | 674.40 | 2025-12-09 09:15:00 | 641.35 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-04 12:30:00 | 675.10 | 2025-12-10 15:15:00 | 640.68 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-12-04 13:30:00 | 674.45 | 2025-12-10 15:15:00 | 640.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 674.70 | 2025-12-10 15:15:00 | 640.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:45:00 | 674.40 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-12-04 12:30:00 | 675.10 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-04 13:30:00 | 674.45 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 674.70 | 2025-12-11 09:15:00 | 647.50 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-12-05 14:45:00 | 664.90 | 2025-12-12 09:15:00 | 655.60 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-12-08 09:45:00 | 663.00 | 2025-12-12 09:15:00 | 655.60 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-01-28 10:15:00 | 650.10 | 2026-01-28 11:15:00 | 657.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-02-17 14:15:00 | 604.25 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-17 15:15:00 | 603.10 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-18 09:30:00 | 603.00 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-18 15:15:00 | 603.35 | 2026-02-19 09:15:00 | 608.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-23 09:15:00 | 609.10 | 2026-02-23 10:15:00 | 602.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-03-04 09:15:00 | 573.10 | 2026-03-06 12:15:00 | 578.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-03-06 10:45:00 | 577.60 | 2026-03-06 12:15:00 | 578.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-04-01 13:30:00 | 530.75 | 2026-04-01 15:15:00 | 531.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-13 12:15:00 | 563.60 | 2026-04-20 15:15:00 | 575.50 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2026-04-15 09:15:00 | 572.35 | 2026-04-20 15:15:00 | 575.50 | STOP_HIT | 1.00 | 0.55% |
