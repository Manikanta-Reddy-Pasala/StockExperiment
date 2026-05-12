# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 645.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 97 |
| ALERT2 | 95 |
| ALERT2_SKIP | 45 |
| ALERT3 | 251 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 123 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 99
- **Target hits / Stop hits / Partials:** 3 / 121 / 8
- **Avg / median % per leg:** 0.08% / -0.65%
- **Sum % (uncompounded):** 10.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 10 | 15.9% | 0 | 63 | 0 | -0.62% | -39.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.35% | -0.4% |
| BUY @ 3rd Alert (retest2) | 62 | 10 | 16.1% | 0 | 62 | 0 | -0.63% | -39.0% |
| SELL (all) | 69 | 23 | 33.3% | 3 | 58 | 8 | 0.72% | 49.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 69 | 23 | 33.3% | 3 | 58 | 8 | 0.72% | 49.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.35% | -0.4% |
| retest2 (combined) | 131 | 33 | 25.2% | 3 | 120 | 8 | 0.08% | 10.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 708.10 | 714.49 | 715.10 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 720.30 | 715.79 | 715.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 12:15:00 | 714.05 | 716.77 | 716.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 712.40 | 714.74 | 715.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 714.00 | 713.02 | 714.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 714.00 | 713.02 | 714.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 714.00 | 713.02 | 714.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 714.00 | 713.02 | 714.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 713.55 | 713.13 | 714.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 714.95 | 713.13 | 714.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 714.45 | 713.39 | 714.41 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 715.70 | 714.77 | 714.74 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 713.00 | 714.47 | 714.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 709.60 | 712.81 | 713.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 709.20 | 709.20 | 710.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 709.20 | 709.20 | 710.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 709.20 | 709.20 | 710.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 706.00 | 709.25 | 709.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:30:00 | 706.80 | 708.69 | 709.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:00:00 | 707.25 | 708.40 | 709.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 703.90 | 697.52 | 697.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 703.90 | 697.52 | 697.41 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 677.85 | 696.86 | 697.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 675.80 | 692.65 | 695.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 693.15 | 686.68 | 690.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 693.15 | 686.68 | 690.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 693.15 | 686.68 | 690.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 693.15 | 686.68 | 690.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 701.50 | 689.64 | 691.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 701.50 | 689.64 | 691.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 699.60 | 691.63 | 692.64 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 697.85 | 694.01 | 693.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 701.60 | 697.04 | 695.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 714.55 | 716.08 | 712.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 714.55 | 716.08 | 712.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 714.55 | 716.08 | 712.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:45:00 | 713.45 | 716.08 | 712.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 713.75 | 715.21 | 712.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 713.75 | 715.21 | 712.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 712.80 | 714.55 | 712.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 712.80 | 714.55 | 712.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 713.95 | 714.43 | 712.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:30:00 | 711.75 | 714.43 | 712.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 712.40 | 714.02 | 712.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:30:00 | 715.95 | 714.40 | 713.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 725.10 | 728.47 | 728.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 725.10 | 728.47 | 728.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 722.75 | 726.90 | 727.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 730.20 | 727.06 | 727.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 730.20 | 727.06 | 727.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 730.20 | 727.06 | 727.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 730.20 | 727.06 | 727.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 730.25 | 727.69 | 727.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 731.25 | 727.69 | 727.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 732.00 | 728.56 | 728.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 736.05 | 731.82 | 730.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 732.05 | 732.14 | 731.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 732.05 | 732.14 | 731.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 732.05 | 732.14 | 731.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 732.05 | 732.14 | 731.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 730.05 | 731.72 | 731.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 729.25 | 731.72 | 731.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 730.50 | 731.48 | 731.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 732.40 | 731.48 | 731.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 722.90 | 729.45 | 730.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 11:15:00 | 722.90 | 729.45 | 730.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 12:15:00 | 722.55 | 728.07 | 729.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 726.50 | 726.38 | 728.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:00:00 | 726.50 | 726.38 | 728.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 727.35 | 726.57 | 728.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 725.40 | 726.07 | 727.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 12:15:00 | 721.40 | 718.39 | 718.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 12:15:00 | 721.40 | 718.39 | 718.27 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 715.40 | 718.55 | 718.74 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 732.35 | 721.31 | 719.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 13:15:00 | 735.00 | 724.05 | 721.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 727.30 | 727.97 | 724.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 727.30 | 727.97 | 724.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 728.10 | 728.51 | 725.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:00:00 | 729.75 | 728.79 | 726.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:00:00 | 730.00 | 729.04 | 727.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 733.70 | 737.69 | 737.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 733.70 | 737.69 | 737.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 732.00 | 736.55 | 737.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 732.00 | 730.92 | 733.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:45:00 | 732.25 | 730.92 | 733.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 733.50 | 731.44 | 733.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 733.50 | 731.44 | 733.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 731.50 | 731.45 | 733.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 729.75 | 731.45 | 733.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 725.05 | 730.17 | 732.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:45:00 | 721.35 | 728.05 | 731.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 13:30:00 | 723.50 | 725.34 | 729.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 724.85 | 723.74 | 727.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 11:00:00 | 720.30 | 723.05 | 726.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 726.60 | 723.51 | 725.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 726.60 | 723.51 | 725.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 728.00 | 724.41 | 726.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 727.35 | 724.41 | 726.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 726.30 | 724.79 | 726.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 724.45 | 724.79 | 726.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 722.05 | 724.24 | 725.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 714.80 | 723.62 | 725.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 716.15 | 722.13 | 724.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 718.05 | 722.12 | 724.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 15:15:00 | 733.20 | 725.58 | 725.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 733.20 | 725.58 | 725.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 735.25 | 728.76 | 727.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 730.35 | 734.12 | 730.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 730.35 | 734.12 | 730.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 730.35 | 734.12 | 730.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 730.35 | 734.12 | 730.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 730.75 | 733.44 | 730.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:45:00 | 732.70 | 733.29 | 730.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:30:00 | 733.25 | 732.26 | 730.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 735.05 | 731.72 | 730.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 721.40 | 730.19 | 730.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 09:15:00 | 721.40 | 730.19 | 730.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 09:15:00 | 705.50 | 721.04 | 725.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 714.90 | 712.22 | 717.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 714.90 | 712.22 | 717.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 714.90 | 712.22 | 717.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 714.30 | 712.22 | 717.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 717.75 | 713.33 | 717.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 717.65 | 713.33 | 717.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 719.50 | 714.56 | 717.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 719.50 | 714.56 | 717.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 722.75 | 716.20 | 718.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 722.75 | 716.20 | 718.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 720.65 | 718.45 | 718.81 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 723.30 | 719.42 | 719.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 13:15:00 | 726.65 | 721.41 | 720.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 721.80 | 723.03 | 721.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 721.80 | 723.03 | 721.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 721.80 | 723.03 | 721.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 721.80 | 723.03 | 721.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 721.00 | 722.63 | 721.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 720.75 | 722.63 | 721.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 719.90 | 722.08 | 721.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 719.90 | 722.08 | 721.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 718.30 | 721.33 | 720.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 718.35 | 721.33 | 720.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 720.45 | 720.78 | 720.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 716.05 | 719.72 | 720.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 705.40 | 705.09 | 709.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 705.40 | 705.09 | 709.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 704.20 | 702.76 | 706.34 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 716.80 | 707.29 | 706.98 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 702.95 | 710.37 | 710.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 13:15:00 | 702.00 | 705.82 | 708.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 10:15:00 | 704.70 | 703.51 | 706.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:00:00 | 704.70 | 703.51 | 706.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 704.70 | 703.75 | 705.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:30:00 | 705.80 | 703.75 | 705.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 696.00 | 693.82 | 696.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 696.00 | 693.82 | 696.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 697.70 | 694.90 | 696.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 697.70 | 694.90 | 696.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 699.20 | 695.76 | 697.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 698.90 | 695.76 | 697.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 701.65 | 697.29 | 697.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 700.35 | 697.29 | 697.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 698.20 | 697.57 | 697.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:30:00 | 699.20 | 697.57 | 697.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 699.05 | 697.87 | 697.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 700.75 | 699.00 | 698.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 14:15:00 | 709.00 | 709.18 | 706.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 15:00:00 | 709.00 | 709.18 | 706.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 730.00 | 734.36 | 730.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 728.00 | 734.36 | 730.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 726.75 | 732.83 | 730.56 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 720.35 | 728.91 | 729.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 717.95 | 726.72 | 728.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 725.30 | 723.92 | 726.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 725.30 | 723.92 | 726.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 725.30 | 723.92 | 726.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 723.50 | 723.92 | 726.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:30:00 | 723.30 | 725.03 | 725.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 723.60 | 724.92 | 725.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 731.50 | 726.33 | 726.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 731.50 | 726.33 | 726.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 11:15:00 | 735.05 | 728.07 | 727.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 766.95 | 766.98 | 762.06 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:15:00 | 794.30 | 766.98 | 762.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 796.70 | 798.61 | 788.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 796.70 | 798.61 | 788.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 797.20 | 798.75 | 791.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 791.50 | 797.30 | 791.94 | SL hit (close<ema400) qty=1.00 sl=791.94 alert=retest1 |

### Cycle 25 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 797.35 | 800.46 | 800.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 795.95 | 799.20 | 800.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 788.65 | 786.70 | 791.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 788.65 | 786.70 | 791.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 788.65 | 786.70 | 791.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 782.00 | 787.27 | 791.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 783.55 | 789.41 | 790.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:30:00 | 783.50 | 786.40 | 788.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 12:15:00 | 794.60 | 790.13 | 789.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 794.60 | 790.13 | 789.96 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 785.50 | 790.02 | 790.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 782.60 | 788.54 | 789.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 775.10 | 774.16 | 778.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 775.10 | 774.16 | 778.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 778.50 | 775.03 | 778.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 778.50 | 775.03 | 778.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 779.75 | 775.97 | 778.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 779.55 | 775.97 | 778.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 778.20 | 776.42 | 778.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:30:00 | 779.80 | 776.42 | 778.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 781.15 | 777.36 | 779.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:00:00 | 781.15 | 777.36 | 779.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 780.40 | 777.97 | 779.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 780.40 | 777.97 | 779.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 788.30 | 780.71 | 780.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 798.15 | 784.19 | 781.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 785.60 | 787.79 | 784.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 785.60 | 787.79 | 784.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 785.60 | 787.79 | 784.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 785.60 | 787.79 | 784.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 787.50 | 787.73 | 784.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 786.30 | 787.73 | 784.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 782.90 | 786.76 | 784.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 782.90 | 786.76 | 784.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 781.70 | 785.75 | 784.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 782.10 | 785.75 | 784.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 779.50 | 783.03 | 783.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 773.70 | 780.74 | 782.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 11:15:00 | 734.70 | 733.46 | 741.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 12:00:00 | 734.70 | 733.46 | 741.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 742.20 | 734.36 | 738.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 742.20 | 734.36 | 738.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 743.20 | 736.12 | 738.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 743.20 | 736.12 | 738.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 737.90 | 738.76 | 739.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:30:00 | 740.25 | 738.76 | 739.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 740.05 | 739.02 | 739.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 740.70 | 739.02 | 739.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 743.75 | 739.96 | 739.90 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 738.30 | 739.68 | 739.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 737.05 | 738.89 | 739.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 10:15:00 | 735.85 | 735.01 | 736.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 10:15:00 | 735.85 | 735.01 | 736.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 735.85 | 735.01 | 736.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:15:00 | 735.00 | 735.01 | 736.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 737.40 | 735.49 | 736.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:00:00 | 737.40 | 735.49 | 736.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 737.00 | 735.79 | 736.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:45:00 | 735.15 | 735.90 | 736.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 739.75 | 736.05 | 736.17 | SL hit (close>static) qty=1.00 sl=738.65 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 740.00 | 736.84 | 736.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 11:15:00 | 741.00 | 737.98 | 737.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 736.15 | 739.47 | 738.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 736.15 | 739.47 | 738.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 736.15 | 739.47 | 738.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 736.15 | 739.47 | 738.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 741.15 | 739.81 | 738.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:45:00 | 743.55 | 740.12 | 739.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 734.15 | 738.59 | 738.58 | SL hit (close<static) qty=1.00 sl=735.10 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 737.35 | 738.34 | 738.46 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 738.80 | 738.59 | 738.56 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 732.50 | 738.01 | 738.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 726.55 | 734.65 | 736.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 710.80 | 710.02 | 717.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 710.80 | 710.02 | 717.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 712.40 | 710.04 | 713.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:00:00 | 712.40 | 710.04 | 713.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 712.00 | 711.04 | 712.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 706.70 | 711.04 | 712.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 13:15:00 | 671.37 | 683.18 | 692.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 13:15:00 | 672.05 | 671.87 | 680.91 | SL hit (close>ema200) qty=0.50 sl=671.87 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 690.00 | 683.86 | 683.69 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 679.70 | 683.01 | 683.38 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 688.95 | 684.20 | 683.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 13:15:00 | 689.85 | 686.25 | 684.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 687.95 | 690.07 | 687.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 10:15:00 | 687.95 | 690.07 | 687.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 687.95 | 690.07 | 687.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 687.95 | 690.07 | 687.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 691.70 | 690.40 | 688.14 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 682.95 | 686.58 | 687.00 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 696.25 | 688.37 | 687.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 699.15 | 692.52 | 689.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 700.45 | 700.51 | 697.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 15:15:00 | 700.40 | 700.56 | 697.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 700.40 | 700.56 | 697.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 702.85 | 700.56 | 697.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 694.90 | 698.52 | 698.09 | SL hit (close<static) qty=1.00 sl=697.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 697.15 | 697.71 | 697.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 692.70 | 696.44 | 697.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 680.75 | 680.46 | 685.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:45:00 | 681.05 | 680.46 | 685.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 687.00 | 681.59 | 685.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:45:00 | 678.85 | 683.30 | 684.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 678.50 | 681.10 | 683.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 678.95 | 680.67 | 682.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 675.25 | 680.86 | 681.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 673.55 | 679.40 | 680.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 691.75 | 681.38 | 680.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 691.75 | 681.38 | 680.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 698.05 | 684.72 | 681.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 698.35 | 698.73 | 693.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 696.45 | 698.73 | 693.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 705.60 | 709.65 | 705.73 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 700.00 | 703.63 | 704.07 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 710.90 | 705.08 | 704.69 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 701.00 | 704.08 | 704.42 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 708.40 | 705.22 | 704.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 711.15 | 706.46 | 705.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 13:15:00 | 718.55 | 720.61 | 717.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 14:00:00 | 718.55 | 720.61 | 717.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 717.35 | 719.96 | 717.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 717.35 | 719.96 | 717.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 718.70 | 719.71 | 717.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 719.30 | 719.71 | 717.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 723.60 | 720.48 | 717.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 734.00 | 720.67 | 719.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 721.55 | 727.17 | 727.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 721.55 | 727.17 | 727.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 717.95 | 725.33 | 726.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 723.20 | 723.13 | 725.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 723.20 | 723.13 | 725.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 726.05 | 723.72 | 725.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 726.05 | 723.72 | 725.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 727.00 | 724.37 | 725.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 728.15 | 724.37 | 725.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 727.25 | 725.85 | 725.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 726.35 | 725.85 | 725.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 727.20 | 726.12 | 726.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 727.20 | 726.12 | 726.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 15:15:00 | 729.10 | 727.56 | 726.81 | Break + close above crossover candle high |

### Cycle 49 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 720.45 | 726.14 | 726.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 719.25 | 722.93 | 724.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 697.50 | 696.07 | 701.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 697.50 | 696.07 | 701.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 704.50 | 695.59 | 698.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 704.50 | 695.59 | 698.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 697.75 | 696.03 | 698.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:45:00 | 696.75 | 696.03 | 698.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 695.30 | 696.13 | 697.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 681.00 | 673.96 | 673.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 681.00 | 673.96 | 673.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 687.00 | 676.57 | 674.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 10:15:00 | 725.35 | 728.28 | 718.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 725.35 | 728.28 | 718.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 727.70 | 731.42 | 724.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 729.35 | 731.42 | 724.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 724.40 | 730.01 | 724.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 723.00 | 730.01 | 724.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 719.40 | 727.89 | 724.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 719.40 | 727.89 | 724.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 723.75 | 727.06 | 724.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 725.00 | 727.06 | 724.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 726.70 | 730.71 | 727.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 12:00:00 | 725.85 | 727.96 | 726.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:00:00 | 726.15 | 728.81 | 727.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 723.60 | 727.77 | 727.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:45:00 | 721.35 | 727.77 | 727.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 726.05 | 728.19 | 727.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 726.05 | 728.19 | 727.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-10 14:15:00 | 722.75 | 727.10 | 727.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 722.75 | 727.10 | 727.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 709.55 | 722.69 | 725.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 724.50 | 716.37 | 719.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 724.50 | 716.37 | 719.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 724.50 | 716.37 | 719.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 724.50 | 716.37 | 719.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 728.10 | 718.72 | 720.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 728.10 | 718.72 | 720.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 730.55 | 722.89 | 721.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 13:15:00 | 732.50 | 724.81 | 722.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 728.10 | 728.49 | 725.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 728.10 | 728.49 | 725.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 732.95 | 729.14 | 726.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:30:00 | 724.70 | 729.14 | 726.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 730.75 | 729.64 | 726.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:45:00 | 728.40 | 729.64 | 726.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 747.30 | 747.56 | 739.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 765.05 | 757.87 | 751.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 13:15:00 | 764.75 | 758.90 | 752.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 766.55 | 760.43 | 753.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 765.00 | 759.32 | 756.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 759.95 | 760.91 | 758.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 762.60 | 760.91 | 758.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 764.00 | 761.53 | 759.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-27 12:15:00 | 756.25 | 758.39 | 758.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 756.25 | 758.39 | 758.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 15:15:00 | 753.05 | 756.40 | 757.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 759.00 | 756.70 | 757.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 10:15:00 | 759.00 | 756.70 | 757.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 759.00 | 756.70 | 757.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 10:45:00 | 756.90 | 756.70 | 757.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 762.80 | 757.92 | 757.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 771.40 | 760.61 | 759.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 14:15:00 | 760.70 | 761.39 | 759.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 14:15:00 | 760.70 | 761.39 | 759.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 760.70 | 761.39 | 759.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:00:00 | 760.70 | 761.39 | 759.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 757.00 | 760.51 | 759.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 750.15 | 760.51 | 759.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 09:15:00 | 750.20 | 758.45 | 758.67 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 766.95 | 760.41 | 759.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 769.00 | 762.63 | 760.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 14:15:00 | 760.45 | 762.19 | 760.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 14:15:00 | 760.45 | 762.19 | 760.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 760.45 | 762.19 | 760.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 760.45 | 762.19 | 760.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 763.05 | 762.36 | 760.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 762.15 | 762.36 | 760.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 764.10 | 762.71 | 761.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 14:45:00 | 772.95 | 768.02 | 764.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 13:15:00 | 811.90 | 818.48 | 818.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 811.90 | 818.48 | 818.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 14:15:00 | 809.95 | 816.78 | 818.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 815.00 | 812.17 | 814.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 815.00 | 812.17 | 814.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 815.00 | 812.17 | 814.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 815.00 | 812.17 | 814.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 816.00 | 812.94 | 814.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 806.40 | 812.94 | 814.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:45:00 | 812.55 | 797.46 | 798.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 816.30 | 801.23 | 799.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 816.30 | 801.23 | 799.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 15:15:00 | 817.60 | 804.50 | 801.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 12:15:00 | 854.80 | 854.95 | 839.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 13:00:00 | 854.80 | 854.95 | 839.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 851.80 | 852.49 | 848.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:30:00 | 854.40 | 852.49 | 848.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 848.45 | 851.68 | 848.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 848.45 | 851.68 | 848.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 850.00 | 851.34 | 848.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 12:15:00 | 854.95 | 851.34 | 848.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 14:30:00 | 852.60 | 856.95 | 854.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 12:15:00 | 847.35 | 852.38 | 852.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 847.35 | 852.38 | 852.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 834.20 | 847.64 | 850.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 837.95 | 834.75 | 839.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 13:00:00 | 837.95 | 834.75 | 839.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 835.90 | 834.98 | 839.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 836.65 | 834.98 | 839.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 840.55 | 836.09 | 839.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 840.55 | 836.09 | 839.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 838.45 | 836.56 | 839.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 832.50 | 836.56 | 839.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 842.55 | 837.76 | 839.51 | SL hit (close>static) qty=1.00 sl=840.80 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 846.90 | 840.53 | 840.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 857.15 | 844.18 | 842.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 840.00 | 851.02 | 847.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 840.00 | 851.02 | 847.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 840.00 | 851.02 | 847.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 840.00 | 851.02 | 847.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 832.00 | 847.21 | 846.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 832.00 | 847.21 | 846.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 832.05 | 844.18 | 845.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 829.10 | 841.17 | 843.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 828.75 | 825.93 | 830.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 828.75 | 825.93 | 830.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 834.60 | 827.66 | 831.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 834.60 | 827.66 | 831.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 834.45 | 829.02 | 831.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 836.30 | 829.02 | 831.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 834.85 | 830.56 | 831.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 833.55 | 830.56 | 831.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 840.00 | 832.45 | 832.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 843.25 | 835.85 | 834.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 840.25 | 840.61 | 837.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 840.25 | 840.61 | 837.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 837.10 | 839.89 | 838.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 837.30 | 839.89 | 838.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 842.75 | 840.46 | 838.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 845.35 | 841.03 | 838.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 12:15:00 | 831.90 | 838.69 | 838.25 | SL hit (close<static) qty=1.00 sl=834.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 834.50 | 837.86 | 837.91 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 847.00 | 838.70 | 838.19 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 15:15:00 | 838.35 | 840.06 | 840.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 13:15:00 | 836.50 | 838.47 | 839.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 833.90 | 833.63 | 835.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 833.90 | 833.63 | 835.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 833.90 | 833.63 | 835.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 839.00 | 833.63 | 835.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 836.90 | 833.67 | 835.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 836.90 | 833.67 | 835.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 836.95 | 834.33 | 835.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:15:00 | 838.35 | 834.33 | 835.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 843.20 | 837.49 | 836.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 845.85 | 840.42 | 838.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 844.20 | 844.66 | 842.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 840.00 | 843.72 | 842.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 840.00 | 843.72 | 842.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 840.00 | 843.72 | 842.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 850.30 | 845.04 | 842.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:00:00 | 855.80 | 847.19 | 843.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 13:15:00 | 853.40 | 847.66 | 844.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:30:00 | 854.70 | 854.56 | 851.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 850.05 | 867.89 | 867.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 850.05 | 867.89 | 867.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 847.70 | 853.99 | 858.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 848.35 | 845.14 | 850.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 15:00:00 | 848.35 | 845.14 | 850.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 836.20 | 820.96 | 830.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 842.65 | 820.96 | 830.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 840.25 | 824.82 | 831.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 840.25 | 824.82 | 831.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 848.90 | 835.71 | 835.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 849.75 | 844.76 | 841.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 907.95 | 909.46 | 900.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 10:15:00 | 899.35 | 907.44 | 900.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 899.35 | 907.44 | 900.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 899.35 | 907.44 | 900.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 903.65 | 906.68 | 900.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:15:00 | 910.20 | 905.95 | 900.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 905.75 | 909.09 | 904.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 875.30 | 912.01 | 911.88 | SL hit (close<static) qty=1.00 sl=896.40 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 865.00 | 902.61 | 907.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 10:15:00 | 861.25 | 873.85 | 887.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 881.00 | 869.30 | 878.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 881.00 | 869.30 | 878.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 881.00 | 869.30 | 878.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 881.05 | 869.30 | 878.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 878.10 | 871.06 | 878.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 875.85 | 872.04 | 878.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 876.75 | 872.98 | 878.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:15:00 | 877.65 | 875.23 | 878.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:45:00 | 877.55 | 875.85 | 878.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 881.55 | 876.99 | 878.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-30 10:15:00 | 881.55 | 876.99 | 878.40 | SL hit (close>static) qty=1.00 sl=881.25 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 890.65 | 880.23 | 879.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 895.30 | 884.35 | 881.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 892.60 | 895.92 | 889.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 892.60 | 895.92 | 889.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 890.50 | 894.27 | 890.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 890.50 | 894.27 | 890.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 891.30 | 893.68 | 890.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 892.65 | 891.97 | 890.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 882.10 | 899.75 | 899.65 | SL hit (close<static) qty=1.00 sl=888.90 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 876.65 | 895.13 | 897.56 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 900.65 | 892.11 | 891.48 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 889.35 | 892.08 | 892.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 15:15:00 | 887.90 | 890.15 | 891.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 898.20 | 891.76 | 891.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 899.00 | 893.21 | 892.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 899.95 | 894.56 | 893.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 910.10 | 911.30 | 907.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 912.60 | 912.99 | 910.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 908.75 | 912.99 | 910.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 904.70 | 911.06 | 909.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 905.20 | 911.06 | 909.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 903.55 | 909.56 | 909.22 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 901.45 | 907.94 | 908.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 898.70 | 904.76 | 906.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 903.70 | 903.21 | 905.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 903.70 | 903.21 | 905.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 904.15 | 903.40 | 905.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:30:00 | 903.60 | 903.40 | 905.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 895.30 | 892.17 | 896.92 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 911.10 | 899.46 | 898.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 914.00 | 902.37 | 900.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 900.55 | 904.95 | 902.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 910.90 | 906.14 | 903.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 914.15 | 907.74 | 904.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 913.30 | 909.88 | 906.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 913.25 | 911.22 | 910.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 14:15:00 | 912.30 | 911.21 | 910.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 914.20 | 911.81 | 910.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 916.50 | 913.58 | 911.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 916.80 | 918.86 | 918.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 910.85 | 917.26 | 917.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 910.85 | 917.26 | 917.91 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 935.25 | 920.49 | 919.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 943.30 | 925.05 | 921.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 1002.90 | 1006.09 | 990.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 1002.90 | 1006.09 | 990.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 993.50 | 999.47 | 995.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 993.00 | 999.47 | 995.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 992.30 | 998.04 | 995.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 992.30 | 998.04 | 995.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 993.20 | 1000.48 | 998.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 996.95 | 999.53 | 998.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 995.85 | 998.37 | 998.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 994.25 | 1002.27 | 1002.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 994.25 | 1002.27 | 1002.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 991.60 | 997.76 | 1000.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 944.60 | 943.12 | 954.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 944.60 | 943.12 | 954.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 948.60 | 944.71 | 953.54 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 974.15 | 956.80 | 956.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 979.60 | 961.36 | 958.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 974.40 | 976.72 | 971.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 974.40 | 976.72 | 971.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 972.95 | 975.97 | 971.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 970.10 | 975.97 | 971.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 961.00 | 972.98 | 970.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 963.40 | 972.98 | 970.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 953.25 | 969.03 | 968.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 953.25 | 969.03 | 968.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 958.20 | 966.86 | 967.93 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 977.00 | 969.30 | 968.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 982.65 | 971.97 | 970.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 957.65 | 969.87 | 969.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 982.40 | 972.38 | 970.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 988.35 | 975.77 | 972.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 960.30 | 970.32 | 971.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 960.30 | 970.32 | 971.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 14:15:00 | 957.20 | 967.70 | 970.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 914.70 | 913.12 | 923.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:30:00 | 915.70 | 913.12 | 923.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 914.00 | 911.89 | 916.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 917.55 | 911.89 | 916.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 913.85 | 910.26 | 912.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 913.85 | 910.26 | 912.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 915.50 | 911.31 | 912.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 916.25 | 911.31 | 912.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 920.30 | 914.83 | 914.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 924.65 | 916.79 | 915.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 931.70 | 932.08 | 926.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 931.70 | 932.08 | 926.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 923.85 | 931.60 | 928.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 922.00 | 931.60 | 928.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 920.30 | 929.34 | 927.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 920.30 | 929.34 | 927.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 919.50 | 925.80 | 926.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 915.00 | 923.64 | 925.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 14:15:00 | 898.00 | 896.15 | 900.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:45:00 | 899.00 | 896.15 | 900.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 898.10 | 896.51 | 900.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 902.40 | 896.51 | 900.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 901.40 | 897.49 | 900.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 901.40 | 897.49 | 900.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 901.50 | 898.29 | 900.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 903.00 | 898.29 | 900.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 905.00 | 900.55 | 901.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:45:00 | 905.15 | 900.55 | 901.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 910.20 | 902.48 | 901.89 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 898.15 | 901.27 | 901.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 894.00 | 898.48 | 899.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 899.80 | 897.52 | 899.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 899.25 | 897.87 | 899.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 896.10 | 897.77 | 898.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 895.40 | 898.47 | 898.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 15:15:00 | 895.50 | 898.37 | 898.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 851.29 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 850.63 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 850.72 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-31 09:15:00 | 806.49 | 815.94 | 828.94 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 798.40 | 794.92 | 794.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 800.00 | 796.70 | 795.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 794.25 | 796.48 | 795.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 792.35 | 795.66 | 795.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 792.35 | 795.66 | 795.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 788.55 | 794.23 | 794.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 787.05 | 791.32 | 793.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 790.60 | 790.33 | 792.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:45:00 | 790.85 | 790.33 | 792.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 790.45 | 790.36 | 791.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 792.70 | 790.36 | 791.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 790.35 | 790.06 | 791.51 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 808.45 | 793.19 | 792.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 812.00 | 799.56 | 795.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 815.55 | 816.18 | 810.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 816.00 | 816.18 | 810.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 820.05 | 823.47 | 821.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 818.05 | 823.47 | 821.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 826.00 | 823.98 | 821.62 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 808.70 | 820.05 | 820.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 803.85 | 816.81 | 819.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 815.65 | 813.85 | 816.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 802.65 | 802.20 | 805.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 802.65 | 802.20 | 805.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 804.10 | 802.58 | 805.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 800.25 | 802.58 | 805.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 800.70 | 802.28 | 805.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 806.05 | 802.21 | 804.25 | SL hit (close>static) qty=1.00 sl=805.90 alert=retest2 |

### Cycle 92 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 815.35 | 807.12 | 806.19 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 799.55 | 804.79 | 805.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 795.80 | 802.99 | 804.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 803.80 | 800.58 | 802.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 797.85 | 800.03 | 802.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 796.20 | 800.03 | 802.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:30:00 | 795.10 | 798.32 | 801.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 805.40 | 798.46 | 800.11 | SL hit (close>static) qty=1.00 sl=803.90 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 801.45 | 796.68 | 796.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 803.00 | 798.64 | 797.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 853.50 | 856.14 | 848.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 860.20 | 856.14 | 848.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 866.15 | 858.14 | 850.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:45:00 | 874.40 | 864.08 | 855.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 880.90 | 886.37 | 886.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 880.90 | 886.37 | 886.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 877.35 | 884.56 | 885.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 872.85 | 871.92 | 876.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:30:00 | 872.55 | 871.92 | 876.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 878.75 | 872.17 | 873.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 880.95 | 872.17 | 873.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 880.30 | 873.80 | 874.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 880.30 | 873.80 | 874.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 880.50 | 875.14 | 874.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 12:15:00 | 886.30 | 877.37 | 876.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 880.00 | 880.18 | 877.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 882.10 | 880.18 | 877.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 886.00 | 881.34 | 878.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 889.30 | 883.81 | 879.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 869.10 | 882.85 | 881.71 | SL hit (close<static) qty=1.00 sl=875.30 alert=retest2 |

### Cycle 97 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 878.80 | 880.98 | 880.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 872.90 | 877.00 | 878.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 878.70 | 877.34 | 878.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 874.80 | 876.83 | 878.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 873.20 | 876.83 | 878.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 874.15 | 871.33 | 874.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 872.65 | 871.60 | 873.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 872.65 | 871.81 | 873.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 872.90 | 872.02 | 873.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 868.20 | 873.97 | 874.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 875.70 | 867.37 | 869.74 | SL hit (close>static) qty=1.00 sl=874.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 875.55 | 871.60 | 871.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 882.50 | 873.78 | 872.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 904.65 | 905.25 | 897.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 904.65 | 905.25 | 897.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 921.30 | 920.25 | 912.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 923.05 | 919.92 | 914.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 13:45:00 | 924.00 | 920.67 | 917.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 924.95 | 921.53 | 918.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 922.45 | 920.95 | 918.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 922.00 | 921.88 | 919.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 923.25 | 921.88 | 919.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 920.95 | 921.69 | 919.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 920.95 | 921.69 | 919.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 920.05 | 921.36 | 919.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 918.65 | 920.81 | 919.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 915.45 | 919.74 | 919.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 915.45 | 919.74 | 919.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 914.55 | 918.70 | 918.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 914.55 | 918.70 | 918.95 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 925.70 | 919.51 | 918.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 933.35 | 922.28 | 920.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 935.00 | 935.51 | 930.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 937.65 | 935.51 | 930.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 929.65 | 933.69 | 931.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 929.65 | 933.69 | 931.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 929.85 | 932.92 | 931.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 926.90 | 932.92 | 931.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 928.00 | 930.96 | 930.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 930.55 | 930.96 | 930.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 927.90 | 936.11 | 936.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 927.90 | 936.11 | 936.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 922.50 | 930.40 | 933.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 930.05 | 926.07 | 930.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 936.45 | 928.15 | 930.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 901.45 | 928.15 | 930.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 873.65 | 870.17 | 870.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 873.65 | 870.17 | 870.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 884.25 | 874.37 | 872.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 887.50 | 879.88 | 877.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 12:15:00 | 870.25 | 876.87 | 877.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 870.25 | 876.87 | 877.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 868.55 | 874.36 | 875.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 863.40 | 863.38 | 868.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:45:00 | 863.05 | 863.38 | 868.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 868.40 | 864.65 | 868.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 868.70 | 864.65 | 868.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 873.15 | 866.35 | 868.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 873.15 | 866.35 | 868.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 873.75 | 867.83 | 869.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 873.75 | 867.83 | 869.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 876.55 | 870.68 | 870.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 878.75 | 873.96 | 872.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 877.60 | 882.33 | 878.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 870.00 | 879.86 | 877.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 870.00 | 879.86 | 877.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 869.20 | 877.73 | 876.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 869.20 | 877.73 | 876.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 872.40 | 875.33 | 875.61 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 877.95 | 875.53 | 875.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 880.00 | 876.42 | 875.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 877.40 | 877.97 | 876.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 880.45 | 878.46 | 876.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 881.30 | 879.27 | 878.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 872.30 | 878.25 | 878.17 | SL hit (close<static) qty=1.00 sl=876.90 alert=retest2 |

### Cycle 107 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 874.95 | 877.59 | 877.88 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 881.35 | 877.84 | 877.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 890.00 | 880.27 | 878.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 13:15:00 | 882.45 | 884.01 | 881.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 882.45 | 884.01 | 881.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 883.95 | 884.00 | 881.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 882.25 | 884.00 | 881.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 883.45 | 883.95 | 881.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 879.75 | 883.95 | 881.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 877.70 | 882.70 | 881.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 877.70 | 882.70 | 881.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 878.30 | 881.82 | 881.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 878.30 | 881.82 | 881.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 874.55 | 880.37 | 880.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 869.40 | 877.64 | 879.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 868.35 | 864.60 | 869.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 877.10 | 867.10 | 870.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 877.10 | 867.10 | 870.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 879.00 | 869.48 | 871.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 879.00 | 869.48 | 871.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 882.90 | 873.85 | 873.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 886.40 | 878.14 | 875.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 878.25 | 878.61 | 876.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 878.25 | 878.61 | 876.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 876.25 | 878.29 | 876.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 876.25 | 878.29 | 876.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 870.10 | 876.65 | 875.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 870.10 | 876.65 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 870.25 | 875.37 | 875.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 867.85 | 875.37 | 875.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 869.90 | 874.27 | 874.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 871.45 | 873.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 14:15:00 | 865.05 | 862.87 | 866.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 865.05 | 862.87 | 866.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 866.00 | 863.49 | 866.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 868.40 | 863.49 | 866.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 865.95 | 863.98 | 866.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 869.25 | 863.98 | 866.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 873.95 | 865.98 | 866.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 874.95 | 865.98 | 866.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 882.55 | 869.29 | 868.24 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 869.20 | 871.33 | 871.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 858.15 | 868.18 | 869.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 846.15 | 839.15 | 846.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 846.15 | 839.15 | 846.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 848.50 | 841.02 | 846.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 849.40 | 841.02 | 846.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 848.70 | 842.56 | 847.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 851.70 | 842.56 | 847.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 848.30 | 844.10 | 847.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 848.30 | 844.10 | 847.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 848.35 | 844.95 | 847.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 855.70 | 844.95 | 847.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 856.50 | 848.64 | 848.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 864.40 | 853.08 | 850.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 871.00 | 871.38 | 864.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 869.60 | 870.84 | 864.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 866.55 | 870.14 | 866.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 863.65 | 870.14 | 866.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 870.00 | 870.11 | 866.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 872.30 | 870.11 | 866.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 873.15 | 870.72 | 867.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 876.25 | 870.72 | 867.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 863.95 | 868.84 | 868.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 863.95 | 868.84 | 868.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 857.15 | 865.89 | 867.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 848.20 | 844.02 | 850.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 848.15 | 844.02 | 850.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 845.35 | 844.29 | 850.14 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 861.30 | 853.65 | 853.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 865.40 | 858.95 | 856.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 859.05 | 859.16 | 856.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:45:00 | 859.75 | 859.16 | 856.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 859.00 | 859.13 | 856.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:30:00 | 856.50 | 859.13 | 856.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 855.85 | 858.47 | 856.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 855.85 | 858.47 | 856.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 860.00 | 858.78 | 857.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 862.75 | 858.78 | 857.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 875.05 | 880.82 | 881.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 875.05 | 880.82 | 881.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 866.05 | 873.91 | 877.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 13:15:00 | 855.70 | 853.77 | 858.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:45:00 | 855.60 | 853.77 | 858.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 859.25 | 854.87 | 858.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 859.25 | 854.87 | 858.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 858.00 | 855.50 | 858.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 854.25 | 854.60 | 858.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 853.90 | 853.64 | 857.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 811.54 | 826.91 | 835.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 811.20 | 826.91 | 835.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 790.00 | 789.49 | 798.15 | SL hit (close>ema200) qty=0.50 sl=789.49 alert=retest2 |

### Cycle 118 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 757.70 | 749.59 | 748.76 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 744.20 | 749.50 | 750.20 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 754.25 | 749.52 | 749.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 755.70 | 751.93 | 750.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 765.00 | 765.61 | 762.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 764.55 | 765.61 | 762.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 759.95 | 764.48 | 762.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 759.95 | 764.48 | 762.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 765.10 | 764.60 | 762.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 768.50 | 765.48 | 763.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 767.50 | 765.98 | 764.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 758.30 | 765.35 | 765.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 758.30 | 765.35 | 765.82 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 775.05 | 766.77 | 765.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 781.95 | 771.82 | 769.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 786.80 | 784.44 | 780.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 786.55 | 784.86 | 780.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 789.80 | 790.05 | 787.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 784.15 | 786.77 | 786.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 784.15 | 786.77 | 786.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 770.60 | 782.92 | 785.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 780.15 | 775.16 | 779.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 783.60 | 776.85 | 780.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 774.30 | 776.85 | 780.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 778.10 | 777.45 | 779.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 778.50 | 777.45 | 779.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 778.45 | 777.51 | 779.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 783.85 | 778.78 | 779.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 784.45 | 778.78 | 779.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 789.35 | 780.89 | 780.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 789.35 | 780.89 | 780.67 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 778.25 | 780.64 | 780.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 772.80 | 779.07 | 780.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 774.90 | 772.88 | 775.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 774.90 | 772.88 | 775.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 782.60 | 774.83 | 776.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 755.95 | 774.83 | 776.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 718.15 | 740.85 | 754.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 730.70 | 728.20 | 737.36 | SL hit (close>ema200) qty=0.50 sl=728.20 alert=retest2 |

### Cycle 126 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 712.20 | 701.70 | 700.38 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 694.30 | 699.26 | 699.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 685.70 | 693.44 | 696.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 673.00 | 691.66 | 694.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 702.25 | 677.82 | 677.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 702.25 | 677.82 | 677.20 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 674.70 | 684.63 | 685.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 662.40 | 678.50 | 682.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 640.50 | 633.93 | 643.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 640.50 | 633.93 | 643.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 636.65 | 633.34 | 636.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 636.65 | 633.34 | 636.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 632.80 | 633.24 | 636.23 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 666.00 | 641.40 | 639.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 681.75 | 670.25 | 662.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 668.35 | 673.85 | 669.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 669.75 | 672.82 | 669.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 669.05 | 671.78 | 668.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:45:00 | 668.00 | 671.32 | 669.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 672.10 | 671.48 | 669.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 688.45 | 671.56 | 669.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 677.80 | 686.29 | 686.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 677.80 | 686.29 | 686.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 674.20 | 683.87 | 685.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 12:15:00 | 681.90 | 680.96 | 683.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 12:30:00 | 680.45 | 680.96 | 683.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 680.95 | 680.96 | 683.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 681.65 | 680.96 | 683.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 678.55 | 679.69 | 682.07 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 689.00 | 683.33 | 683.29 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 682.00 | 683.84 | 684.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 672.30 | 680.82 | 682.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 677.20 | 673.65 | 677.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 679.80 | 674.88 | 677.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 679.80 | 674.88 | 677.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 675.60 | 675.03 | 677.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 670.85 | 675.50 | 677.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 637.31 | 646.26 | 654.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 15:15:00 | 645.00 | 644.26 | 650.21 | SL hit (close>ema200) qty=0.50 sl=644.26 alert=retest2 |

### Cycle 134 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 649.90 | 646.44 | 646.20 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 645.55 | 646.61 | 646.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 645.20 | 646.12 | 646.40 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-27 09:45:00 | 706.00 | 2024-06-03 10:15:00 | 703.90 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-05-27 10:30:00 | 706.80 | 2024-06-03 10:15:00 | 703.90 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-05-27 12:00:00 | 707.25 | 2024-06-03 10:15:00 | 703.90 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-06-12 09:30:00 | 715.95 | 2024-06-21 14:15:00 | 725.10 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2024-06-28 09:15:00 | 732.40 | 2024-06-28 11:15:00 | 722.90 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-01 11:30:00 | 725.40 | 2024-07-04 12:15:00 | 721.40 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-07-09 15:00:00 | 729.75 | 2024-07-16 10:15:00 | 733.70 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-07-10 13:00:00 | 730.00 | 2024-07-16 10:15:00 | 733.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-07-19 10:45:00 | 721.35 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-19 13:30:00 | 723.50 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-22 10:15:00 | 724.85 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-22 11:00:00 | 720.30 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-07-23 12:15:00 | 714.80 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-07-23 13:00:00 | 716.15 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-07-23 13:30:00 | 718.05 | 2024-07-23 15:15:00 | 733.20 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-07-25 11:45:00 | 732.70 | 2024-07-26 09:15:00 | 721.40 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-07-25 13:30:00 | 733.25 | 2024-07-26 09:15:00 | 721.40 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-07-25 15:15:00 | 735.05 | 2024-07-26 09:15:00 | 721.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-08-30 10:15:00 | 723.50 | 2024-09-02 10:15:00 | 731.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-08-30 14:30:00 | 723.30 | 2024-09-02 10:15:00 | 731.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-02 09:15:00 | 723.60 | 2024-09-02 10:15:00 | 731.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest1 | 2024-09-06 09:15:00 | 794.30 | 2024-09-10 11:15:00 | 791.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-10 13:15:00 | 799.95 | 2024-09-17 09:15:00 | 797.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-09-11 09:30:00 | 801.80 | 2024-09-17 09:15:00 | 797.35 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-11 11:15:00 | 800.00 | 2024-09-17 09:15:00 | 797.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-09-12 09:15:00 | 804.20 | 2024-09-17 09:15:00 | 797.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-09-19 11:15:00 | 782.00 | 2024-09-23 12:15:00 | 794.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-09-20 13:30:00 | 783.55 | 2024-09-23 12:15:00 | 794.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-23 09:30:00 | 783.50 | 2024-09-23 12:15:00 | 794.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-15 09:45:00 | 735.15 | 2024-10-15 14:15:00 | 739.75 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-10-15 14:45:00 | 734.95 | 2024-10-15 15:15:00 | 740.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-10-17 12:45:00 | 743.55 | 2024-10-18 09:15:00 | 734.15 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-10-25 09:15:00 | 706.70 | 2024-10-28 13:15:00 | 671.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 706.70 | 2024-10-29 13:15:00 | 672.05 | STOP_HIT | 0.50 | 4.90% |
| BUY | retest2 | 2024-11-08 09:15:00 | 702.85 | 2024-11-11 09:15:00 | 694.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-11-18 09:45:00 | 678.85 | 2024-11-25 09:15:00 | 691.75 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-18 12:45:00 | 678.50 | 2024-11-25 09:15:00 | 691.75 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-11-18 14:00:00 | 678.95 | 2024-11-25 09:15:00 | 691.75 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-11-21 09:15:00 | 675.25 | 2024-11-25 09:15:00 | 691.75 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-12-10 10:15:00 | 734.00 | 2024-12-13 09:15:00 | 721.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-16 11:15:00 | 726.35 | 2024-12-16 11:15:00 | 727.20 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-12-24 11:45:00 | 696.75 | 2025-01-02 09:15:00 | 681.00 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest2 | 2024-12-24 15:15:00 | 695.30 | 2025-01-02 09:15:00 | 681.00 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2025-01-08 13:15:00 | 725.00 | 2025-01-10 14:15:00 | 722.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-01-09 09:30:00 | 726.70 | 2025-01-10 14:15:00 | 722.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-01-09 12:00:00 | 725.85 | 2025-01-10 14:15:00 | 722.75 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-01-10 10:00:00 | 726.15 | 2025-01-10 14:15:00 | 722.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-01-21 11:30:00 | 765.05 | 2025-01-27 12:15:00 | 756.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-21 13:15:00 | 764.75 | 2025-01-27 12:15:00 | 756.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-01-21 14:00:00 | 766.55 | 2025-01-27 12:15:00 | 756.25 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-01-23 12:15:00 | 765.00 | 2025-01-27 12:15:00 | 756.25 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-01-30 14:45:00 | 772.95 | 2025-02-06 13:15:00 | 811.90 | STOP_HIT | 1.00 | 5.04% |
| SELL | retest2 | 2025-02-10 09:15:00 | 806.40 | 2025-02-12 14:15:00 | 816.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-12 13:45:00 | 812.55 | 2025-02-12 14:15:00 | 816.30 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-02-18 12:15:00 | 854.95 | 2025-02-20 12:15:00 | 847.35 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-02-19 14:30:00 | 852.60 | 2025-02-20 12:15:00 | 847.35 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-02-25 09:15:00 | 832.50 | 2025-02-25 09:15:00 | 842.55 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-03-07 10:30:00 | 845.35 | 2025-03-07 12:15:00 | 831.90 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-20 12:00:00 | 855.80 | 2025-04-01 10:15:00 | 850.05 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-03-20 13:15:00 | 853.40 | 2025-04-01 10:15:00 | 850.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-03-24 10:30:00 | 854.70 | 2025-04-01 10:15:00 | 850.05 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-04-22 13:15:00 | 910.20 | 2025-04-25 09:15:00 | 875.30 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2025-04-23 11:00:00 | 905.75 | 2025-04-25 09:15:00 | 875.30 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-04-29 11:45:00 | 875.85 | 2025-04-30 10:15:00 | 881.55 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-04-29 13:00:00 | 876.75 | 2025-04-30 10:15:00 | 881.55 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-04-29 15:15:00 | 877.65 | 2025-04-30 10:15:00 | 881.55 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-04-30 09:45:00 | 877.55 | 2025-04-30 10:15:00 | 881.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-04-30 15:15:00 | 869.15 | 2025-05-02 09:15:00 | 887.10 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-05-07 09:15:00 | 892.65 | 2025-05-08 15:15:00 | 882.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-27 12:00:00 | 914.15 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-05-27 14:00:00 | 913.30 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-05-29 11:45:00 | 913.25 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-29 14:15:00 | 912.30 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-30 09:30:00 | 916.50 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-03 11:45:00 | 916.80 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-13 11:30:00 | 996.95 | 2025-06-17 11:15:00 | 994.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-13 13:15:00 | 995.85 | 2025-06-17 11:15:00 | 994.25 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-06-27 15:00:00 | 988.35 | 2025-06-30 13:15:00 | 960.30 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-07-21 14:00:00 | 896.10 | 2025-07-28 09:15:00 | 851.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 895.40 | 2025-07-28 09:15:00 | 850.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 15:15:00 | 895.50 | 2025-07-28 09:15:00 | 850.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:00:00 | 896.10 | 2025-07-31 09:15:00 | 806.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 895.40 | 2025-07-31 09:15:00 | 805.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 15:15:00 | 895.50 | 2025-07-31 09:15:00 | 805.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-01 09:15:00 | 800.25 | 2025-09-01 13:15:00 | 806.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-01 11:15:00 | 800.70 | 2025-09-01 13:15:00 | 806.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-03 11:15:00 | 796.20 | 2025-09-04 09:15:00 | 805.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-03 12:30:00 | 795.10 | 2025-09-04 09:15:00 | 805.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-04 12:15:00 | 796.10 | 2025-09-08 12:15:00 | 801.45 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-04 13:30:00 | 795.65 | 2025-09-08 12:15:00 | 801.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-15 12:45:00 | 874.40 | 2025-09-19 11:15:00 | 880.90 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-09-25 10:30:00 | 889.30 | 2025-09-26 09:15:00 | 869.10 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-29 11:15:00 | 873.20 | 2025-10-01 14:15:00 | 875.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-09-30 11:15:00 | 874.15 | 2025-10-03 10:15:00 | 874.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-09-30 12:00:00 | 872.65 | 2025-10-03 10:15:00 | 874.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-09-30 13:00:00 | 872.65 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-01 09:15:00 | 868.20 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-03 09:15:00 | 869.75 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-03 10:15:00 | 870.00 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-09 15:00:00 | 923.05 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-10 13:45:00 | 924.00 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-10 15:00:00 | 924.95 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-13 10:00:00 | 922.45 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-20 09:15:00 | 930.55 | 2025-10-23 15:15:00 | 927.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-27 09:15:00 | 901.45 | 2025-11-13 09:15:00 | 873.65 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2025-11-17 13:30:00 | 887.50 | 2025-11-18 12:15:00 | 870.25 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-11-28 11:45:00 | 881.30 | 2025-12-01 09:15:00 | 872.30 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-12-24 10:15:00 | 876.25 | 2025-12-26 14:15:00 | 863.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-01-02 09:15:00 | 862.75 | 2026-01-08 11:15:00 | 875.05 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2026-01-14 09:30:00 | 854.25 | 2026-01-20 13:15:00 | 811.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:30:00 | 853.90 | 2026-01-20 13:15:00 | 811.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 854.25 | 2026-01-23 10:15:00 | 790.00 | STOP_HIT | 0.50 | 7.52% |
| SELL | retest2 | 2026-01-14 11:30:00 | 853.90 | 2026-01-23 10:15:00 | 790.00 | STOP_HIT | 0.50 | 7.48% |
| BUY | retest2 | 2026-02-11 14:45:00 | 768.50 | 2026-02-13 12:15:00 | 758.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-12 10:15:00 | 767.50 | 2026-02-13 12:15:00 | 758.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-19 12:15:00 | 786.80 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-02-19 13:00:00 | 786.55 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-23 09:15:00 | 789.80 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-02-25 09:15:00 | 774.30 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-25 10:30:00 | 778.10 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-25 11:00:00 | 778.50 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-25 12:45:00 | 778.45 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-02 09:15:00 | 755.95 | 2026-03-04 11:15:00 | 718.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 755.95 | 2026-03-05 14:15:00 | 730.70 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-03-23 09:15:00 | 673.00 | 2026-03-25 09:15:00 | 702.25 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 668.35 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2026-04-13 10:45:00 | 669.75 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2026-04-13 12:15:00 | 669.05 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2026-04-13 12:45:00 | 668.00 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2026-04-15 09:15:00 | 688.45 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-27 14:30:00 | 670.85 | 2026-04-30 10:15:00 | 637.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 670.85 | 2026-04-30 15:15:00 | 645.00 | STOP_HIT | 0.50 | 3.85% |
