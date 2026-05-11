# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 905.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 9 |
| TARGET_HIT | 11 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 11
- **Target hits / Stop hits / Partials:** 11 / 16 / 9
- **Avg / median % per leg:** 3.83% / 5.00%
- **Sum % (uncompounded):** 137.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 7 | 3 | 0 | 6.87% | 68.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 8 | 80.0% | 7 | 3 | 0 | 6.87% | 68.7% |
| SELL (all) | 26 | 17 | 65.4% | 4 | 13 | 9 | 2.66% | 69.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 17 | 65.4% | 4 | 13 | 9 | 2.66% | 69.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 25 | 69.4% | 11 | 16 | 9 | 3.83% | 137.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 15:15:00 | 644.90 | 614.91 | 614.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 647.55 | 615.23 | 615.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 617.45 | 618.30 | 616.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 11:00:00 | 617.45 | 618.30 | 616.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 619.65 | 618.32 | 616.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:30:00 | 618.50 | 618.32 | 616.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 614.90 | 618.50 | 616.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 614.90 | 618.50 | 616.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 621.30 | 618.53 | 616.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 625.20 | 618.40 | 616.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 608.00 | 618.36 | 616.93 | SL hit (close<static) qty=1.00 sl=612.85 alert=retest2 |

### Cycle 2 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 554.00 | 615.09 | 615.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 546.00 | 614.40 | 614.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 613.35 | 611.29 | 613.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 613.35 | 611.29 | 613.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 613.35 | 611.29 | 613.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 609.00 | 611.29 | 613.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 612.70 | 611.30 | 613.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:00:00 | 612.70 | 611.30 | 613.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 612.80 | 611.32 | 613.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 12:45:00 | 611.95 | 611.32 | 613.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 13:45:00 | 611.60 | 611.32 | 613.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 14:15:00 | 631.10 | 611.52 | 613.36 | SL hit (close>static) qty=1.00 sl=613.75 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 640.80 | 615.27 | 615.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 645.95 | 616.31 | 615.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 655.70 | 675.44 | 655.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 655.70 | 675.44 | 655.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 655.70 | 675.44 | 655.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 655.70 | 675.44 | 655.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 653.95 | 675.23 | 655.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 653.95 | 675.23 | 655.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 659.15 | 675.07 | 655.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:45:00 | 662.00 | 674.92 | 655.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:30:00 | 664.00 | 674.58 | 656.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-30 10:15:00 | 728.20 | 679.29 | 662.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 798.00 | 832.40 | 832.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 790.85 | 831.99 | 832.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 789.00 | 786.19 | 803.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 789.00 | 786.19 | 803.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 779.25 | 785.62 | 802.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:30:00 | 789.00 | 785.62 | 802.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 788.90 | 777.19 | 794.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 751.00 | 783.60 | 795.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 713.45 | 779.56 | 793.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 764.35 | 773.36 | 788.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 776.25 | 773.39 | 788.71 | SL hit (close>ema200) qty=0.50 sl=773.39 alert=retest2 |

### Cycle 5 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 850.50 | 770.15 | 770.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 12:15:00 | 855.00 | 770.99 | 770.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 791.80 | 825.16 | 802.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 791.80 | 825.16 | 802.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 791.80 | 825.16 | 802.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 818.00 | 824.02 | 802.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 10:45:00 | 815.00 | 824.20 | 803.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:15:00 | 812.75 | 824.20 | 803.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-16 10:15:00 | 894.03 | 829.94 | 808.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 839.00 | 854.20 | 854.22 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 866.20 | 854.24 | 854.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 10:15:00 | 880.35 | 857.06 | 855.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 862.10 | 862.17 | 858.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:45:00 | 861.30 | 862.17 | 858.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 859.00 | 862.12 | 858.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 855.35 | 862.12 | 858.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 854.00 | 862.04 | 858.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:30:00 | 863.95 | 861.83 | 858.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 849.00 | 861.55 | 858.67 | SL hit (close<static) qty=1.00 sl=852.60 alert=retest2 |

### Cycle 8 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 906.10 | 919.93 | 919.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 919.15 | 919.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 835.45 | 822.51 | 848.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 844.60 | 823.75 | 844.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 844.60 | 823.75 | 844.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 840.80 | 823.75 | 844.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 845.05 | 823.97 | 844.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 845.05 | 823.97 | 844.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 845.95 | 824.19 | 844.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 845.65 | 824.19 | 844.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 846.00 | 824.40 | 844.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 846.35 | 824.40 | 844.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 846.10 | 824.62 | 844.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 850.00 | 824.62 | 844.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 847.90 | 824.85 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 847.90 | 824.85 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 845.00 | 825.05 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 842.50 | 825.05 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 845.20 | 825.25 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 847.30 | 825.25 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 844.00 | 825.44 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 843.20 | 825.44 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 838.50 | 825.57 | 844.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 837.00 | 829.23 | 844.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 836.65 | 829.41 | 844.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:30:00 | 834.40 | 829.54 | 844.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 795.15 | 826.12 | 841.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 794.82 | 826.12 | 841.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 823.95 | 823.82 | 839.03 | SL hit (close>ema200) qty=0.50 sl=823.82 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 903.25 | 839.90 | 839.74 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 791.90 | 844.78 | 845.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 765.35 | 843.48 | 844.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 809.60 | 808.03 | 822.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:00:00 | 809.60 | 808.03 | 822.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 832.00 | 800.88 | 815.82 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 884.20 | 825.60 | 825.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 890.00 | 827.44 | 826.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 13:15:00 | 618.50 | 2024-05-18 09:15:00 | 632.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-05-17 15:15:00 | 618.00 | 2024-05-18 09:15:00 | 632.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-06-03 14:00:00 | 625.20 | 2024-06-04 09:15:00 | 608.00 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-06-07 12:45:00 | 611.95 | 2024-06-07 14:15:00 | 631.10 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-06-07 13:45:00 | 611.60 | 2024-06-07 14:15:00 | 631.10 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-07-19 12:45:00 | 662.00 | 2024-07-30 10:15:00 | 728.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 09:30:00 | 664.00 | 2024-07-30 12:15:00 | 730.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 751.00 | 2025-01-28 09:15:00 | 713.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 751.00 | 2025-01-30 12:15:00 | 776.25 | STOP_HIT | 0.50 | -3.36% |
| SELL | retest2 | 2025-01-30 11:30:00 | 764.35 | 2025-02-07 12:15:00 | 787.75 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-01-30 13:30:00 | 774.00 | 2025-02-07 12:15:00 | 787.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-01-30 14:15:00 | 772.95 | 2025-02-07 12:15:00 | 787.75 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-02-06 12:30:00 | 775.65 | 2025-02-07 12:15:00 | 787.75 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-02-06 13:45:00 | 773.20 | 2025-02-11 10:15:00 | 735.30 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-02-07 09:15:00 | 770.80 | 2025-02-11 11:15:00 | 734.30 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-02-07 11:15:00 | 770.00 | 2025-02-11 11:15:00 | 730.74 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-02-10 09:15:00 | 769.20 | 2025-02-12 09:15:00 | 726.13 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-02-06 13:45:00 | 773.20 | 2025-02-18 11:15:00 | 696.60 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-02-07 09:15:00 | 770.80 | 2025-02-18 11:15:00 | 695.66 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2025-02-07 11:15:00 | 770.00 | 2025-02-18 11:15:00 | 692.28 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2025-02-10 09:15:00 | 769.20 | 2025-02-19 14:15:00 | 687.92 | TARGET_HIT | 0.50 | 10.57% |
| BUY | retest2 | 2025-04-07 15:15:00 | 818.00 | 2025-04-16 10:15:00 | 894.03 | TARGET_HIT | 1.00 | 9.29% |
| BUY | retest2 | 2025-04-09 10:45:00 | 815.00 | 2025-04-16 13:15:00 | 899.80 | TARGET_HIT | 1.00 | 10.40% |
| BUY | retest2 | 2025-04-09 11:15:00 | 812.75 | 2025-04-16 13:15:00 | 896.50 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2025-06-19 13:00:00 | 812.80 | 2025-06-26 09:15:00 | 839.00 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-07-10 14:30:00 | 863.95 | 2025-07-11 10:15:00 | 849.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-07-15 09:15:00 | 862.90 | 2025-07-24 09:15:00 | 949.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 10:30:00 | 863.05 | 2025-07-24 09:15:00 | 949.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 837.00 | 2026-01-12 09:15:00 | 795.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 836.65 | 2026-01-12 09:15:00 | 794.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 837.00 | 2026-01-14 10:15:00 | 823.95 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2026-01-06 13:15:00 | 836.65 | 2026-01-14 10:15:00 | 823.95 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-01-06 14:30:00 | 834.40 | 2026-01-21 11:15:00 | 792.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 832.55 | 2026-01-21 11:15:00 | 790.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:30:00 | 834.40 | 2026-01-28 10:15:00 | 821.50 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-01-16 09:15:00 | 832.55 | 2026-01-28 10:15:00 | 821.50 | STOP_HIT | 0.50 | 1.33% |
