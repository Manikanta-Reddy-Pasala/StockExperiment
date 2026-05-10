# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 835.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 10 |
| TARGET_HIT | 9 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 18
- **Target hits / Stop hits / Partials:** 9 / 20 / 10
- **Avg / median % per leg:** 2.42% / 0.40%
- **Sum % (uncompounded):** 94.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| SELL (all) | 37 | 21 | 56.8% | 9 | 18 | 10 | 2.71% | 100.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 21 | 56.8% | 9 | 18 | 10 | 2.71% | 100.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 21 | 53.8% | 9 | 20 | 10 | 2.42% | 94.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 787.00 | 758.30 | 758.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 795.34 | 759.90 | 759.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 794.42 | 803.90 | 785.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:00:00 | 807.12 | 803.15 | 786.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:00:00 | 804.82 | 807.63 | 790.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 781.54 | 832.42 | 819.53 | SL hit (close<static) qty=1.00 sl=785.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 781.54 | 832.42 | 819.53 | SL hit (close<static) qty=1.00 sl=785.20 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 750.46 | 808.67 | 808.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 743.04 | 796.23 | 802.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 772.68 | 772.09 | 783.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:30:00 | 774.38 | 772.09 | 783.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 780.82 | 772.47 | 782.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 776.70 | 772.70 | 782.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 791.42 | 773.01 | 781.78 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 776.24 | 782.81 | 785.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:30:00 | 776.18 | 782.54 | 785.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 776.40 | 774.78 | 780.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 778.78 | 772.38 | 778.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 782.00 | 772.38 | 778.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 768.74 | 772.34 | 778.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 765.36 | 772.34 | 778.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 765.56 | 771.26 | 777.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 764.40 | 771.15 | 777.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 764.08 | 771.03 | 776.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 776.80 | 770.35 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 776.80 | 770.35 | 776.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 775.24 | 770.40 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 776.30 | 770.40 | 776.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 774.00 | 770.43 | 776.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 774.80 | 770.43 | 776.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 773.62 | 770.46 | 776.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 777.20 | 770.46 | 776.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 779.60 | 770.60 | 776.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 779.34 | 770.60 | 776.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 778.44 | 770.68 | 776.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 776.44 | 770.89 | 776.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=781.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=781.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=781.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=781.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 789.14 | 771.18 | 776.21 | SL hit (close>static) qty=1.00 sl=782.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 761.36 | 774.15 | 777.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 787.62 | 773.90 | 777.17 | SL hit (close>static) qty=1.00 sl=782.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 773.96 | 776.09 | 778.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 735.26 | 773.02 | 776.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 796.06 | 770.87 | 774.82 | SL hit (close>ema200) qty=0.50 sl=770.87 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 751.50 | 778.44 | 778.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 746.00 | 777.38 | 778.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 767.70 | 764.28 | 770.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 767.70 | 764.28 | 770.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 771.10 | 764.09 | 769.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 772.70 | 764.09 | 769.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 771.50 | 764.17 | 769.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 773.80 | 764.17 | 769.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 757.40 | 756.29 | 763.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:15:00 | 744.90 | 756.04 | 763.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 745.00 | 755.84 | 763.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.65 | 752.81 | 761.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.75 | 752.81 | 761.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 670.41 | 724.86 | 741.26 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 670.50 | 724.86 | 741.26 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 745.45 | 724.72 | 737.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 746.60 | 725.35 | 737.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 736.10 | 725.97 | 737.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 733.80 | 725.97 | 737.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:00:00 | 733.20 | 726.09 | 737.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 733.80 | 726.16 | 737.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:00:00 | 733.70 | 726.18 | 736.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 737.05 | 726.29 | 736.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 737.05 | 726.29 | 736.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 738.00 | 726.41 | 736.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 740.85 | 726.41 | 736.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 743.60 | 726.58 | 736.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 743.60 | 726.58 | 736.47 | SL hit (close>static) qty=1.00 sl=738.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 743.60 | 726.58 | 736.47 | SL hit (close>static) qty=1.00 sl=738.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 743.60 | 726.58 | 736.47 | SL hit (close>static) qty=1.00 sl=738.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 743.60 | 726.58 | 736.47 | SL hit (close>static) qty=1.00 sl=738.70 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-17 10:15:00 | 743.85 | 726.58 | 736.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 732.60 | 726.64 | 736.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:00:00 | 732.45 | 726.70 | 736.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 732.30 | 726.92 | 736.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 731.50 | 727.08 | 736.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 730.95 | 727.08 | 736.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 736.45 | 727.45 | 736.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 736.45 | 727.45 | 736.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 734.95 | 727.52 | 736.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 732.95 | 727.52 | 736.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 733.30 | 727.58 | 736.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 733.30 | 727.58 | 736.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 732.75 | 727.63 | 736.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 730.00 | 727.65 | 736.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 709.27 | 726.15 | 734.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 708.18 | 725.96 | 734.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 695.83 | 723.07 | 732.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 695.68 | 723.07 | 732.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 694.92 | 723.07 | 732.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 694.40 | 723.07 | 732.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 693.50 | 723.07 | 732.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 670.91 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 671.94 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 659.21 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 659.07 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 658.35 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 657.86 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 657.00 | 720.09 | 730.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 731.10 | 677.89 | 689.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 731.60 | 678.97 | 689.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 738.90 | 680.62 | 690.56 | SL hit (close>static) qty=1.00 sl=736.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 738.90 | 680.62 | 690.56 | SL hit (close>static) qty=1.00 sl=736.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 731.50 | 681.80 | 691.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 744.10 | 682.42 | 691.32 | SL hit (close>static) qty=1.00 sl=736.80 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 774.20 | 699.61 | 699.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 779.45 | 720.36 | 711.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 13:00:00 | 807.12 | 2025-07-28 12:15:00 | 781.54 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-06-19 14:00:00 | 804.82 | 2025-07-28 12:15:00 | 781.54 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-09-11 12:45:00 | 776.70 | 2025-09-17 11:15:00 | 791.42 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-24 14:45:00 | 776.24 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-25 14:30:00 | 776.18 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-10-07 10:00:00 | 776.40 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-14 11:15:00 | 765.36 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-16 14:45:00 | 765.56 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-10-17 10:00:00 | 764.40 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-10-17 13:30:00 | 764.08 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-10-24 13:30:00 | 776.44 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-29 09:15:00 | 761.36 | 2025-10-30 09:15:00 | 787.62 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-11-04 09:15:00 | 773.96 | 2025-11-07 09:15:00 | 735.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 773.96 | 2025-11-12 09:15:00 | 796.06 | STOP_HIT | 0.50 | -2.86% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-01-12 11:15:00 | 707.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-01-12 11:15:00 | 707.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-02-01 12:15:00 | 670.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-02-01 12:15:00 | 670.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 10:45:00 | 745.45 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-02-11 14:15:00 | 746.60 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-02-12 11:15:00 | 733.80 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-12 13:00:00 | 733.20 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-02-24 11:15:00 | 709.27 | PARTIAL | 0.50 | 3.34% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-02-24 12:15:00 | 708.18 | PARTIAL | 0.50 | 3.48% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-02-27 09:15:00 | 695.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-02-27 09:15:00 | 695.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-02-27 09:15:00 | 694.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-02-27 09:15:00 | 694.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-02-27 09:15:00 | 693.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-03-02 09:15:00 | 670.91 | TARGET_HIT | 0.50 | 8.57% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-03-02 09:15:00 | 671.94 | TARGET_HIT | 0.50 | 8.42% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-03-02 09:15:00 | 659.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-03-02 09:15:00 | 659.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-03-02 09:15:00 | 658.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-03-02 09:15:00 | 657.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-03-02 09:15:00 | 657.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 09:30:00 | 731.10 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-16 11:30:00 | 731.60 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 731.50 | 2026-04-17 10:15:00 | 744.10 | STOP_HIT | 1.00 | -1.72% |
