# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 459.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 84 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 14 |
| TARGET_HIT | 3 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 36
- **Target hits / Stop hits / Partials:** 3 / 51 / 14
- **Avg / median % per leg:** 0.85% / -0.72%
- **Sum % (uncompounded):** 58.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 2 | 11 | 0 | -0.75% | -9.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 2 | 11 | 0 | -0.75% | -9.7% |
| SELL (all) | 55 | 30 | 54.5% | 1 | 40 | 14 | 1.23% | 67.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 30 | 54.5% | 1 | 40 | 14 | 1.23% | 67.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 32 | 47.1% | 3 | 51 | 14 | 0.85% | 58.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 714.25 | 782.11 | 782.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 710.70 | 781.40 | 781.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 733.45 | 726.78 | 744.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-17 10:00:00 | 733.45 | 726.78 | 744.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 737.85 | 727.05 | 742.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 13:30:00 | 741.45 | 727.05 | 742.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 755.00 | 727.53 | 742.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 755.00 | 727.53 | 742.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 754.35 | 727.80 | 742.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 12:15:00 | 748.70 | 728.03 | 742.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 13:30:00 | 748.55 | 729.86 | 742.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 747.10 | 730.27 | 742.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 10:00:00 | 745.30 | 730.42 | 742.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 748.00 | 731.04 | 743.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:00:00 | 748.00 | 731.04 | 743.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 748.35 | 731.21 | 743.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 748.35 | 731.21 | 743.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 746.45 | 731.36 | 743.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 745.30 | 731.36 | 743.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 745.95 | 731.64 | 743.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 747.60 | 731.64 | 743.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 753.30 | 731.86 | 743.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:00:00 | 753.30 | 731.86 | 743.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 744.95 | 734.07 | 743.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 11:15:00 | 742.35 | 734.07 | 743.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 11:15:00 | 739.55 | 734.12 | 743.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 12:15:00 | 738.65 | 734.12 | 743.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 13:00:00 | 738.95 | 734.17 | 743.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 15:15:00 | 746.55 | 734.47 | 743.53 | SL hit (close>static) qty=1.00 sl=745.25 alert=retest2 |

### Cycle 2 — BUY (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 11:15:00 | 744.20 | 723.60 | 723.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 13:15:00 | 749.25 | 724.07 | 723.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 11:15:00 | 737.55 | 738.21 | 732.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 12:00:00 | 737.55 | 738.21 | 732.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 732.90 | 738.12 | 732.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:30:00 | 733.85 | 738.12 | 732.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 731.10 | 738.05 | 732.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 15:00:00 | 731.10 | 738.05 | 732.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 729.45 | 737.97 | 732.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 729.90 | 737.97 | 732.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 734.05 | 737.86 | 732.28 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 687.15 | 727.70 | 727.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 683.05 | 725.80 | 726.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 707.45 | 701.31 | 711.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:00:00 | 707.45 | 701.31 | 711.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 713.60 | 701.79 | 711.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:00:00 | 713.60 | 701.79 | 711.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 11:15:00 | 714.25 | 701.92 | 711.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:00:00 | 714.25 | 701.92 | 711.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 714.55 | 702.04 | 711.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 14:45:00 | 712.85 | 702.26 | 711.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 12:00:00 | 710.05 | 702.51 | 711.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-02 10:15:00 | 724.90 | 703.26 | 711.31 | SL hit (close>static) qty=1.00 sl=715.85 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 710.30 | 696.87 | 696.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 712.50 | 697.21 | 697.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 875.05 | 877.86 | 838.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 875.05 | 877.86 | 838.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 854.35 | 880.35 | 858.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 855.45 | 880.35 | 858.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 854.15 | 880.09 | 858.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 853.35 | 880.09 | 858.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 842.15 | 879.45 | 858.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:00:00 | 842.15 | 879.45 | 858.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 861.90 | 875.38 | 858.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 870.05 | 875.38 | 858.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 852.40 | 874.75 | 859.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 852.40 | 874.75 | 859.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 860.50 | 874.60 | 859.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 871.10 | 870.75 | 858.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 880.45 | 870.15 | 859.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 845.45 | 876.29 | 865.18 | SL hit (close<static) qty=1.00 sl=851.30 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 849.80 | 873.38 | 873.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 843.55 | 869.86 | 871.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 874.95 | 869.36 | 871.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 874.95 | 869.36 | 871.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 874.95 | 869.36 | 871.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 871.15 | 869.36 | 871.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 884.00 | 869.50 | 871.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 884.00 | 869.50 | 871.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 875.70 | 870.28 | 871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:15:00 | 875.65 | 870.28 | 871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 872.30 | 870.33 | 871.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 872.30 | 870.33 | 871.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 872.85 | 870.36 | 871.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 872.85 | 870.36 | 871.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 869.85 | 870.35 | 871.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 871.55 | 870.35 | 871.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 869.80 | 870.35 | 871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:15:00 | 873.10 | 870.35 | 871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 874.80 | 870.39 | 871.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 874.80 | 870.39 | 871.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 877.20 | 870.46 | 871.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:00:00 | 877.20 | 870.46 | 871.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 870.85 | 870.86 | 871.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 867.60 | 870.84 | 871.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 867.55 | 870.84 | 871.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:15:00 | 824.22 | 869.63 | 871.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:15:00 | 824.17 | 869.63 | 871.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 856.45 | 849.06 | 858.85 | SL hit (close>ema200) qty=0.50 sl=849.06 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 709.50 | 660.11 | 659.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 711.45 | 660.62 | 660.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 674.25 | 674.81 | 668.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 11:00:00 | 674.25 | 674.81 | 668.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 664.75 | 674.59 | 668.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 664.75 | 674.59 | 668.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 670.40 | 674.55 | 668.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 671.65 | 674.55 | 668.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 671.50 | 674.50 | 668.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 654.80 | 673.95 | 668.27 | SL hit (close<static) qty=1.00 sl=664.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 632.50 | 664.96 | 664.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 629.70 | 664.60 | 664.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 663.35 | 658.91 | 661.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 658.65 | 658.91 | 661.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 653.80 | 658.90 | 661.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 652.80 | 658.78 | 661.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 672.45 | 657.51 | 660.22 | SL hit (close>static) qty=1.00 sl=665.25 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 657.00 | 645.96 | 645.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 661.65 | 648.28 | 647.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 646.60 | 650.26 | 648.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 642.55 | 650.18 | 648.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 642.55 | 650.18 | 648.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 625.45 | 646.65 | 646.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 618.45 | 641.88 | 644.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 413.80 | 413.60 | 450.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:45:00 | 413.40 | 413.60 | 450.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 447.25 | 417.99 | 448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 445.55 | 417.99 | 448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 449.15 | 418.30 | 448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 449.70 | 418.30 | 448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 445.40 | 418.57 | 448.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:30:00 | 443.00 | 419.08 | 448.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 449.80 | 419.63 | 448.61 | SL hit (close>static) qty=1.00 sl=449.55 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-24 12:15:00 | 748.70 | 2023-12-04 15:15:00 | 746.55 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2023-11-28 13:30:00 | 748.55 | 2023-12-04 15:15:00 | 746.55 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2023-11-29 09:15:00 | 747.10 | 2023-12-11 14:15:00 | 711.26 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2023-11-29 10:00:00 | 745.30 | 2023-12-11 14:15:00 | 711.12 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2023-12-04 12:15:00 | 738.65 | 2023-12-12 09:15:00 | 709.75 | PARTIAL | 0.50 | 3.91% |
| SELL | retest2 | 2023-12-04 13:00:00 | 738.95 | 2023-12-12 09:15:00 | 708.03 | PARTIAL | 0.50 | 4.18% |
| SELL | retest2 | 2023-12-05 10:00:00 | 738.90 | 2023-12-12 14:15:00 | 701.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-29 09:15:00 | 747.10 | 2024-01-02 09:15:00 | 714.40 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2023-11-29 10:00:00 | 745.30 | 2024-01-02 09:15:00 | 714.40 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2023-12-04 12:15:00 | 738.65 | 2024-01-02 09:15:00 | 714.40 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2023-12-04 13:00:00 | 738.95 | 2024-01-02 09:15:00 | 714.40 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2023-12-05 10:00:00 | 738.90 | 2024-01-02 09:15:00 | 714.40 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2024-01-11 11:30:00 | 739.30 | 2024-01-17 09:15:00 | 702.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 11:30:00 | 739.30 | 2024-01-29 09:15:00 | 665.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-27 14:45:00 | 712.85 | 2024-04-02 10:15:00 | 724.90 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-28 12:00:00 | 710.05 | 2024-04-02 10:15:00 | 724.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-04-16 10:15:00 | 711.35 | 2024-04-19 11:15:00 | 717.65 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-04-16 14:15:00 | 711.55 | 2024-04-19 11:15:00 | 717.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-04-19 14:45:00 | 714.75 | 2024-04-22 09:15:00 | 724.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-04-22 12:00:00 | 714.15 | 2024-05-03 11:15:00 | 678.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-22 12:00:00 | 714.15 | 2024-05-17 10:15:00 | 691.00 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2024-06-10 14:00:00 | 713.70 | 2024-06-27 11:15:00 | 710.30 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-06-19 10:00:00 | 705.20 | 2024-06-27 11:15:00 | 710.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-11-05 09:15:00 | 871.10 | 2024-11-18 09:15:00 | 845.45 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-11-06 09:15:00 | 880.45 | 2024-11-18 09:15:00 | 845.45 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2024-11-22 14:15:00 | 869.95 | 2024-12-02 10:15:00 | 956.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 14:45:00 | 869.65 | 2024-12-02 10:15:00 | 956.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-11 10:30:00 | 911.55 | 2024-12-19 09:15:00 | 851.25 | STOP_HIT | 1.00 | -6.62% |
| SELL | retest2 | 2025-01-10 12:30:00 | 867.60 | 2025-01-13 10:15:00 | 824.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:15:00 | 867.55 | 2025-01-13 10:15:00 | 824.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:30:00 | 867.60 | 2025-01-23 14:15:00 | 856.45 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-01-10 13:15:00 | 867.55 | 2025-01-23 14:15:00 | 856.45 | STOP_HIT | 0.50 | 1.28% |
| BUY | retest2 | 2025-08-07 12:15:00 | 671.65 | 2025-08-08 14:15:00 | 654.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-08-07 14:00:00 | 671.50 | 2025-08-08 14:15:00 | 654.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-08-14 09:15:00 | 672.00 | 2025-08-14 09:15:00 | 663.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-18 09:15:00 | 671.85 | 2025-08-18 14:15:00 | 663.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-22 13:45:00 | 669.80 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-25 09:45:00 | 670.70 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-25 10:15:00 | 671.20 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-25 13:45:00 | 670.05 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-04 15:15:00 | 653.80 | 2025-09-15 11:15:00 | 672.45 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-09-05 10:45:00 | 652.80 | 2025-09-15 11:15:00 | 672.45 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-09-16 13:15:00 | 654.70 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-16 14:00:00 | 654.00 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-17 14:45:00 | 650.25 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-09-18 09:45:00 | 652.60 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-18 10:30:00 | 652.70 | 2025-09-19 14:15:00 | 665.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-09-18 11:00:00 | 652.80 | 2025-09-19 14:15:00 | 665.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-22 13:30:00 | 657.50 | 2025-09-26 09:15:00 | 624.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 657.50 | 2025-10-09 11:15:00 | 643.20 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-10-24 13:15:00 | 657.50 | 2025-10-27 11:15:00 | 665.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 11:30:00 | 657.90 | 2025-11-06 10:15:00 | 625.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:30:00 | 657.95 | 2025-11-06 10:15:00 | 625.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 14:15:00 | 645.50 | 2025-11-07 09:15:00 | 613.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:30:00 | 657.90 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2025-10-29 09:30:00 | 657.95 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2025-11-04 14:15:00 | 645.50 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | -1.44% |
| SELL | retest2 | 2025-11-06 09:15:00 | 640.10 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-11-11 11:15:00 | 645.70 | 2025-11-12 14:15:00 | 663.95 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-11-12 10:45:00 | 644.90 | 2025-11-12 14:15:00 | 663.95 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-11-27 09:30:00 | 644.80 | 2025-11-28 12:15:00 | 647.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-28 12:15:00 | 643.30 | 2025-11-28 12:15:00 | 647.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-01 09:30:00 | 644.55 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-01 12:45:00 | 645.15 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-02 09:15:00 | 640.90 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-21 13:30:00 | 443.00 | 2026-04-21 15:15:00 | 449.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-22 09:15:00 | 440.80 | 2026-04-24 11:15:00 | 418.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 440.80 | 2026-04-27 09:15:00 | 433.45 | STOP_HIT | 0.50 | 1.67% |
