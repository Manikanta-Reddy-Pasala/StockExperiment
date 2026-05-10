# Indian Hotels Co. Ltd. (INDHOTEL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 672.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 25
- **Target hits / Stop hits / Partials:** 2 / 30 / 6
- **Avg / median % per leg:** 0.59% / -0.76%
- **Sum % (uncompounded):** 22.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 13 | 34.2% | 2 | 30 | 6 | 0.59% | 22.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 13 | 34.2% | 2 | 30 | 6 | 0.59% | 22.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 13 | 34.2% | 2 | 30 | 6 | 0.59% | 22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 762.50 | 785.02 | 785.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 755.45 | 779.85 | 782.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 779.85 | 777.33 | 780.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 779.85 | 777.33 | 780.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 779.85 | 777.33 | 780.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 779.85 | 777.33 | 780.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 784.00 | 774.38 | 778.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 784.00 | 774.38 | 778.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 785.40 | 774.49 | 778.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 785.45 | 774.49 | 778.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 777.90 | 774.56 | 778.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:30:00 | 778.10 | 774.56 | 778.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 774.30 | 774.59 | 778.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 778.95 | 774.59 | 778.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 777.00 | 773.85 | 777.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 777.35 | 773.85 | 777.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 774.65 | 773.80 | 777.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 778.45 | 773.80 | 777.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 777.40 | 773.83 | 777.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 777.40 | 773.83 | 777.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 777.30 | 773.87 | 777.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:00:00 | 777.30 | 773.87 | 777.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 776.85 | 773.90 | 777.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 774.55 | 773.92 | 777.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 778.45 | 773.95 | 777.59 | SL hit (close>static) qty=1.00 sl=778.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:45:00 | 774.60 | 773.95 | 777.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 13:15:00 | 781.85 | 774.03 | 777.59 | SL hit (close>static) qty=1.00 sl=778.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 774.50 | 774.84 | 777.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:00:00 | 775.15 | 774.84 | 777.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 735.77 | 772.56 | 776.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 736.39 | 772.56 | 776.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 766.55 | 766.48 | 772.36 | SL hit (close>ema200) qty=0.50 sl=766.48 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 766.55 | 766.48 | 772.36 | SL hit (close>ema200) qty=0.50 sl=766.48 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 774.55 | 766.22 | 771.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 774.55 | 766.22 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 771.75 | 766.28 | 771.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 769.10 | 766.33 | 771.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 780.65 | 766.37 | 771.85 | SL hit (close>static) qty=1.00 sl=775.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 769.50 | 768.18 | 772.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 765.00 | 768.20 | 772.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 13:15:00 | 731.02 | 756.19 | 764.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 14:15:00 | 726.75 | 755.88 | 763.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 757.00 | 753.99 | 762.32 | SL hit (close>ema200) qty=0.50 sl=753.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 757.00 | 753.99 | 762.32 | SL hit (close>ema200) qty=0.50 sl=753.99 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 767.20 | 756.07 | 762.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 764.10 | 756.24 | 762.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 764.05 | 756.24 | 762.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 756.65 | 756.25 | 762.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 752.55 | 756.25 | 762.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 754.50 | 755.86 | 762.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 754.80 | 755.83 | 761.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 754.10 | 755.79 | 761.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 752.00 | 749.19 | 755.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 747.30 | 749.24 | 755.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 763.60 | 749.37 | 755.64 | SL hit (close>static) qty=1.00 sl=758.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 765.05 | 749.52 | 755.69 | SL hit (close>static) qty=1.00 sl=764.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 765.05 | 749.52 | 755.69 | SL hit (close>static) qty=1.00 sl=764.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 765.05 | 749.52 | 755.69 | SL hit (close>static) qty=1.00 sl=764.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 765.05 | 749.52 | 755.69 | SL hit (close>static) qty=1.00 sl=764.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 781.70 | 752.14 | 756.65 | SL hit (close>static) qty=1.00 sl=775.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 807.85 | 760.98 | 760.76 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 709.30 | 765.47 | 765.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 708.85 | 742.01 | 748.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 728.25 | 726.81 | 737.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 728.25 | 726.81 | 737.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 737.30 | 726.93 | 737.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 737.30 | 726.93 | 737.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 736.40 | 727.03 | 737.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 735.30 | 727.12 | 737.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:00:00 | 732.85 | 727.18 | 737.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:45:00 | 733.90 | 727.28 | 737.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 735.30 | 727.72 | 735.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 737.15 | 727.81 | 735.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 737.15 | 727.81 | 735.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 734.95 | 727.88 | 735.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 744.20 | 728.23 | 735.94 | SL hit (close>static) qty=1.00 sl=741.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 744.20 | 728.23 | 735.94 | SL hit (close>static) qty=1.00 sl=741.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 744.20 | 728.23 | 735.94 | SL hit (close>static) qty=1.00 sl=741.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 744.20 | 728.23 | 735.94 | SL hit (close>static) qty=1.00 sl=741.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:45:00 | 733.00 | 731.47 | 736.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 733.70 | 731.61 | 736.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 733.40 | 731.66 | 736.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:15:00 | 733.65 | 731.69 | 736.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 738.70 | 731.70 | 736.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 738.70 | 731.70 | 736.64 | SL hit (close>static) qty=1.00 sl=737.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 738.70 | 731.70 | 736.64 | SL hit (close>static) qty=1.00 sl=737.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 738.70 | 731.70 | 736.64 | SL hit (close>static) qty=1.00 sl=737.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 738.70 | 731.70 | 736.64 | SL hit (close>static) qty=1.00 sl=737.85 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 738.20 | 731.70 | 736.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 733.50 | 731.71 | 736.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 732.65 | 731.71 | 736.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 732.80 | 731.69 | 736.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:00:00 | 732.65 | 730.15 | 734.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 15:15:00 | 733.30 | 730.31 | 734.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 733.30 | 730.34 | 734.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 731.75 | 730.34 | 734.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 733.35 | 730.37 | 734.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 725.25 | 730.50 | 734.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:00:00 | 727.65 | 730.47 | 734.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 727.50 | 730.45 | 734.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 727.40 | 730.43 | 734.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 736.45 | 728.68 | 733.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 736.45 | 728.68 | 733.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 738.40 | 728.78 | 733.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 738.40 | 728.78 | 733.35 | SL hit (close>static) qty=1.00 sl=737.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 738.40 | 728.78 | 733.35 | SL hit (close>static) qty=1.00 sl=737.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 738.40 | 728.78 | 733.35 | SL hit (close>static) qty=1.00 sl=737.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 738.40 | 728.78 | 733.35 | SL hit (close>static) qty=1.00 sl=737.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 738.40 | 728.78 | 733.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 739.55 | 728.89 | 733.38 | SL hit (close>static) qty=1.00 sl=739.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 739.55 | 728.89 | 733.38 | SL hit (close>static) qty=1.00 sl=739.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 739.55 | 728.89 | 733.38 | SL hit (close>static) qty=1.00 sl=739.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 739.55 | 728.89 | 733.38 | SL hit (close>static) qty=1.00 sl=739.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 732.00 | 731.53 | 734.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 736.80 | 731.53 | 734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 734.45 | 731.56 | 734.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 731.45 | 732.02 | 734.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 739.80 | 732.15 | 734.20 | SL hit (close>static) qty=1.00 sl=738.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 731.30 | 733.78 | 734.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 731.45 | 733.76 | 734.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 694.73 | 729.89 | 732.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 694.88 | 729.89 | 732.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 658.17 | 713.88 | 723.45 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 658.31 | 713.88 | 723.45 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-09 09:15:00 | 774.55 | 2025-06-09 11:15:00 | 778.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-06-09 12:45:00 | 774.60 | 2025-06-09 13:15:00 | 781.85 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-06-11 12:30:00 | 774.50 | 2025-06-13 09:15:00 | 735.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 13:00:00 | 775.15 | 2025-06-13 09:15:00 | 736.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:30:00 | 774.50 | 2025-06-20 14:15:00 | 766.55 | STOP_HIT | 0.50 | 1.03% |
| SELL | retest2 | 2025-06-11 13:00:00 | 775.15 | 2025-06-20 14:15:00 | 766.55 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-06-24 12:30:00 | 769.10 | 2025-06-25 09:15:00 | 780.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-06-27 14:30:00 | 769.50 | 2025-07-14 13:15:00 | 731.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 09:15:00 | 765.00 | 2025-07-14 14:15:00 | 726.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-27 14:30:00 | 769.50 | 2025-07-17 09:15:00 | 757.00 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2025-06-30 09:15:00 | 765.00 | 2025-07-17 09:15:00 | 757.00 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2025-07-22 09:45:00 | 767.20 | 2025-08-13 09:15:00 | 763.60 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-07-23 09:15:00 | 752.55 | 2025-08-13 10:15:00 | 765.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-07-24 09:45:00 | 754.50 | 2025-08-13 10:15:00 | 765.05 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-07-24 10:30:00 | 754.80 | 2025-08-13 10:15:00 | 765.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-07-24 12:45:00 | 754.10 | 2025-08-13 10:15:00 | 765.05 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-12 15:00:00 | 747.30 | 2025-08-18 09:15:00 | 781.70 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-11-20 14:15:00 | 735.30 | 2025-11-28 10:15:00 | 744.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-20 15:00:00 | 732.85 | 2025-11-28 10:15:00 | 744.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-21 09:45:00 | 733.90 | 2025-11-28 10:15:00 | 744.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-27 13:00:00 | 735.30 | 2025-11-28 10:15:00 | 744.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-03 11:45:00 | 733.00 | 2025-12-05 10:15:00 | 738.70 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-03 14:45:00 | 733.70 | 2025-12-05 10:15:00 | 738.70 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-12-04 09:30:00 | 733.40 | 2025-12-05 10:15:00 | 738.70 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-04 11:15:00 | 733.65 | 2025-12-05 10:15:00 | 738.70 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-05 12:15:00 | 732.65 | 2025-12-22 10:15:00 | 738.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-08 10:00:00 | 732.80 | 2025-12-22 10:15:00 | 738.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-12 12:00:00 | 732.65 | 2025-12-22 10:15:00 | 738.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-12 15:15:00 | 733.30 | 2025-12-22 10:15:00 | 738.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-16 09:15:00 | 725.25 | 2025-12-22 11:15:00 | 739.55 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-12-16 10:00:00 | 727.65 | 2025-12-22 11:15:00 | 739.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-12-16 10:30:00 | 727.50 | 2025-12-22 11:15:00 | 739.55 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-12-16 11:30:00 | 727.40 | 2025-12-22 11:15:00 | 739.55 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-01 13:15:00 | 731.45 | 2026-01-01 15:15:00 | 739.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-06 12:15:00 | 731.30 | 2026-01-09 09:15:00 | 694.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:00:00 | 731.45 | 2026-01-09 09:15:00 | 694.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 731.30 | 2026-01-20 09:15:00 | 658.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 13:00:00 | 731.45 | 2026-01-20 09:15:00 | 658.31 | TARGET_HIT | 0.50 | 10.00% |
