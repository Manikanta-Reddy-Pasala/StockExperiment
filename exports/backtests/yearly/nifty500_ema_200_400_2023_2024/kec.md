# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 597.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 18
- **Target hits / Stop hits / Partials:** 6 / 18 / 6
- **Avg / median % per leg:** 2.09% / -0.60%
- **Sum % (uncompounded):** 62.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.35% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.35% | -5.6% |
| SELL (all) | 14 | 11 | 78.6% | 5 | 3 | 6 | 4.88% | 68.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 11 | 78.6% | 5 | 3 | 6 | 4.88% | 68.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 12 | 40.0% | 6 | 18 | 6 | 2.09% | 62.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 583.00 | 640.21 | 640.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 15:15:00 | 579.00 | 639.60 | 639.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 604.00 | 600.16 | 614.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 609.95 | 600.83 | 613.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 609.95 | 600.83 | 613.90 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 636.75 | 612.90 | 612.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 655.45 | 614.54 | 613.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 687.20 | 688.87 | 664.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 14:15:00 | 663.00 | 687.83 | 665.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 663.00 | 687.83 | 665.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 760.90 | 712.05 | 688.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 12:15:00 | 836.99 | 763.81 | 742.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 919.65 | 1063.13 | 1063.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 901.10 | 1061.51 | 1062.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 760.50 | 750.69 | 824.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 760.50 | 750.69 | 824.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 835.95 | 752.04 | 823.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 835.95 | 752.04 | 823.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 836.45 | 752.88 | 823.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 836.45 | 752.88 | 823.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 828.90 | 755.99 | 823.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:45:00 | 830.15 | 755.99 | 823.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 798.75 | 762.88 | 824.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 792.55 | 765.90 | 823.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 792.50 | 768.59 | 818.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 788.25 | 769.61 | 816.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:15:00 | 752.92 | 769.37 | 814.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:15:00 | 752.88 | 769.37 | 814.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:15:00 | 748.84 | 769.37 | 814.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-04 09:15:00 | 713.29 | 767.07 | 811.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 875.50 | 773.73 | 773.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 881.00 | 799.74 | 787.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 880.05 | 881.81 | 851.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 880.05 | 881.81 | 851.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 857.05 | 879.79 | 859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 854.85 | 879.79 | 859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 854.55 | 879.53 | 859.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 858.60 | 878.84 | 859.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 13:00:00 | 858.15 | 876.94 | 860.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:30:00 | 861.90 | 876.57 | 860.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:15:00 | 857.85 | 876.16 | 860.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 861.50 | 876.01 | 860.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 865.10 | 875.91 | 860.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:45:00 | 869.20 | 875.84 | 860.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 865.05 | 875.28 | 860.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 853.45 | 874.35 | 861.10 | SL hit (close<static) qty=1.00 sl=856.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 797.55 | 851.03 | 851.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 794.75 | 849.94 | 850.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 844.15 | 841.52 | 846.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 844.15 | 841.52 | 846.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 860.60 | 841.71 | 846.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 860.60 | 841.71 | 846.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 850.35 | 841.79 | 846.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 843.40 | 841.79 | 846.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 801.23 | 832.99 | 840.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 855.00 | 830.51 | 838.63 | SL hit (close>ema200) qty=0.50 sl=830.51 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 880.80 | 844.63 | 844.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 891.15 | 846.27 | 845.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 857.00 | 861.08 | 854.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 857.00 | 861.08 | 854.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 857.00 | 861.08 | 854.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 858.50 | 861.08 | 854.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 852.20 | 860.94 | 854.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 852.15 | 860.94 | 854.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 851.75 | 860.85 | 854.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 850.35 | 860.85 | 854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 854.10 | 858.77 | 853.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 856.50 | 858.74 | 853.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 850.00 | 858.77 | 853.67 | SL hit (close<static) qty=1.00 sl=851.25 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 827.60 | 852.04 | 852.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 826.70 | 851.09 | 851.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 749.35 | 722.10 | 756.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 747.10 | 728.77 | 752.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 742.30 | 729.06 | 752.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 742.25 | 731.12 | 751.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 705.18 | 730.00 | 749.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 705.14 | 730.00 | 749.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 668.07 | 724.67 | 745.78 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 760.90 | 2024-06-12 12:15:00 | 836.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-25 11:15:00 | 792.55 | 2025-04-03 11:15:00 | 752.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 792.50 | 2025-04-03 11:15:00 | 752.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 15:00:00 | 788.25 | 2025-04-03 11:15:00 | 748.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 11:15:00 | 792.55 | 2025-04-04 09:15:00 | 713.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 792.50 | 2025-04-04 09:15:00 | 713.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 15:00:00 | 788.25 | 2025-04-04 09:15:00 | 709.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-16 09:45:00 | 789.40 | 2025-05-26 13:15:00 | 852.45 | STOP_HIT | 1.00 | -7.99% |
| BUY | retest2 | 2025-07-23 15:00:00 | 858.60 | 2025-08-01 09:15:00 | 853.45 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-28 13:00:00 | 858.15 | 2025-08-01 09:15:00 | 853.45 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-28 14:30:00 | 861.90 | 2025-08-01 09:15:00 | 853.45 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-29 11:15:00 | 857.85 | 2025-08-01 11:15:00 | 850.75 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-29 13:00:00 | 865.10 | 2025-08-01 11:15:00 | 850.75 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-29 13:45:00 | 869.20 | 2025-08-01 11:15:00 | 850.75 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-30 10:30:00 | 865.05 | 2025-08-01 11:15:00 | 850.75 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-18 14:15:00 | 843.40 | 2025-08-29 09:15:00 | 801.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 14:15:00 | 843.40 | 2025-09-02 09:15:00 | 855.00 | STOP_HIT | 0.50 | -1.38% |
| SELL | retest2 | 2025-09-04 10:30:00 | 845.40 | 2025-09-09 09:15:00 | 865.05 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-09-30 13:00:00 | 856.50 | 2025-10-01 10:15:00 | 850.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-01 14:15:00 | 855.15 | 2025-10-03 10:15:00 | 849.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-06 09:15:00 | 856.05 | 2025-10-10 14:15:00 | 853.60 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-06 12:30:00 | 860.80 | 2025-10-14 15:15:00 | 852.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-10 11:30:00 | 861.25 | 2025-10-14 15:15:00 | 852.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-13 13:00:00 | 861.45 | 2025-10-14 15:15:00 | 852.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-13 13:30:00 | 860.30 | 2025-10-15 15:15:00 | 850.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-13 14:00:00 | 860.35 | 2025-10-15 15:15:00 | 850.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-02 11:45:00 | 742.30 | 2026-01-08 09:15:00 | 705.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:30:00 | 742.25 | 2026-01-08 09:15:00 | 705.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 11:45:00 | 742.30 | 2026-01-12 09:15:00 | 668.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 09:30:00 | 742.25 | 2026-01-12 09:15:00 | 668.02 | TARGET_HIT | 0.50 | 10.00% |
