# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 5
- **Target hits / Stop hits / Partials:** 6 / 8 / 9
- **Avg / median % per leg:** 4.46% / 5.00%
- **Sum % (uncompounded):** 102.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.93% | -3.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.93% | -3.7% |
| SELL (all) | 19 | 18 | 94.7% | 6 | 4 | 9 | 5.60% | 106.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 18 | 94.7% | 6 | 4 | 9 | 5.60% | 106.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 18 | 78.3% | 6 | 8 | 9 | 4.46% | 102.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 746.90 | 796.98 | 797.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 742.30 | 778.50 | 786.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 732.75 | 729.40 | 748.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 732.15 | 729.40 | 748.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 741.65 | 713.35 | 730.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 741.65 | 713.35 | 730.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 740.25 | 713.62 | 730.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 744.50 | 713.62 | 730.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 764.90 | 742.97 | 742.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 766.45 | 743.21 | 743.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 745.45 | 745.48 | 744.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 742.05 | 746.00 | 744.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 742.05 | 746.00 | 744.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 740.95 | 745.95 | 744.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 741.70 | 745.95 | 744.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 741.90 | 745.83 | 744.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 746.15 | 745.83 | 744.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 770.00 | 789.62 | 774.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 768.70 | 789.62 | 774.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 771.50 | 789.44 | 774.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 770.50 | 789.44 | 774.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 771.25 | 786.27 | 774.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 771.25 | 786.27 | 774.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 774.90 | 786.16 | 774.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 771.10 | 786.16 | 774.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 768.90 | 785.99 | 774.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 768.90 | 785.99 | 774.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 769.90 | 785.83 | 774.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 774.50 | 785.47 | 774.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 765.50 | 785.12 | 773.98 | SL hit (close<static) qty=1.00 sl=768.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 775.00 | 782.14 | 773.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 766.65 | 781.98 | 773.20 | SL hit (close<static) qty=1.00 sl=768.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 775.00 | 780.91 | 772.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 774.15 | 780.23 | 772.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 776.00 | 785.22 | 777.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 776.00 | 785.22 | 777.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 778.00 | 785.15 | 777.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 768.80 | 784.51 | 777.18 | SL hit (close<static) qty=1.00 sl=768.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 768.80 | 784.51 | 777.18 | SL hit (close<static) qty=1.00 sl=768.90 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 739.20 | 772.47 | 772.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 725.55 | 771.67 | 772.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 747.10 | 739.83 | 751.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 751.40 | 739.94 | 751.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 746.70 | 740.06 | 751.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 747.95 | 739.67 | 751.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 748.00 | 739.85 | 751.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 747.50 | 740.12 | 750.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 751.00 | 740.23 | 750.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 748.70 | 740.23 | 750.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 753.00 | 740.35 | 750.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 753.00 | 740.35 | 750.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 747.15 | 740.42 | 750.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 756.80 | 740.42 | 750.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 750.00 | 740.59 | 750.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 750.15 | 740.59 | 750.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 750.05 | 740.69 | 750.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:45:00 | 750.00 | 740.69 | 750.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 748.00 | 740.76 | 750.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:30:00 | 751.20 | 740.76 | 750.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 709.37 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.55 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.60 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.12 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 12:15:00 | 673.16 | 731.81 | 744.07 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 12:15:00 | 673.20 | 731.81 | 744.07 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 672.03 | 731.23 | 743.72 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 672.75 | 731.23 | 743.72 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 741.65 | 710.71 | 729.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 742.35 | 710.71 | 729.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 737.20 | 710.98 | 729.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 738.95 | 710.98 | 729.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 720.00 | 711.30 | 729.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 709.85 | 711.30 | 729.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 717.90 | 707.06 | 723.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 716.15 | 707.23 | 723.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:30:00 | 717.65 | 707.69 | 722.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 719.65 | 707.81 | 722.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 719.65 | 707.81 | 722.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 682.00 | 704.62 | 717.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 681.77 | 704.62 | 717.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 15:15:00 | 680.34 | 704.36 | 717.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 707.00 | 703.31 | 716.33 | SL hit (close>ema200) qty=0.50 sl=703.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 707.00 | 703.31 | 716.33 | SL hit (close>ema200) qty=0.50 sl=703.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 707.00 | 703.31 | 716.33 | SL hit (close>ema200) qty=0.50 sl=703.31 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 705.75 | 703.33 | 716.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 738.85 | 703.33 | 716.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 715.30 | 703.45 | 716.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 715.30 | 703.45 | 716.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 715.00 | 703.56 | 716.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 709.55 | 703.63 | 716.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 722.55 | 703.90 | 716.06 | SL hit (close>static) qty=1.00 sl=720.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 713.70 | 704.82 | 716.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 674.36 | 703.45 | 714.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 678.01 | 703.45 | 714.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 15:15:00 | 642.33 | 698.76 | 711.59 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 638.87 | 687.26 | 703.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 699.90 | 672.79 | 672.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 703.90 | 673.89 | 673.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-04 09:15:00 | 774.50 | 2025-11-04 10:15:00 | 765.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-07 11:30:00 | 775.00 | 2025-11-07 12:15:00 | 766.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-10 12:45:00 | 775.00 | 2025-11-21 10:15:00 | 768.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-11 13:00:00 | 774.15 | 2025-11-21 10:15:00 | 768.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-05 12:15:00 | 746.70 | 2026-01-16 14:15:00 | 709.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:30:00 | 747.95 | 2026-01-16 14:15:00 | 710.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:45:00 | 748.00 | 2026-01-16 14:15:00 | 710.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:00:00 | 747.50 | 2026-01-16 14:15:00 | 710.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 12:15:00 | 746.70 | 2026-01-20 12:15:00 | 673.16 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2026-01-06 11:30:00 | 747.95 | 2026-01-20 12:15:00 | 673.20 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2026-01-06 14:45:00 | 748.00 | 2026-01-20 13:15:00 | 672.03 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-01-07 15:00:00 | 747.50 | 2026-01-20 13:15:00 | 672.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 709.85 | 2026-02-19 14:15:00 | 682.00 | PARTIAL | 0.50 | 3.92% |
| SELL | retest2 | 2026-02-09 14:30:00 | 717.90 | 2026-02-19 14:15:00 | 681.77 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-02-10 09:15:00 | 716.15 | 2026-02-19 15:15:00 | 680.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 709.85 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2026-02-09 14:30:00 | 717.90 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-02-10 09:15:00 | 716.15 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2026-02-10 13:30:00 | 717.65 | 2026-02-25 10:15:00 | 722.55 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-24 13:30:00 | 709.55 | 2026-03-02 09:15:00 | 674.36 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-02-26 09:45:00 | 713.70 | 2026-03-02 09:15:00 | 678.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:30:00 | 709.55 | 2026-03-04 15:15:00 | 642.33 | TARGET_HIT | 0.50 | 9.47% |
| SELL | retest2 | 2026-02-26 09:45:00 | 713.70 | 2026-03-11 11:15:00 | 638.87 | TARGET_HIT | 0.50 | 10.49% |
