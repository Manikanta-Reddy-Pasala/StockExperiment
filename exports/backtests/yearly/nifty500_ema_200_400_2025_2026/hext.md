# Hexaware Technologies Ltd. (HEXT)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 486.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 13
- **Target hits / Stop hits / Partials:** 6 / 15 / 7
- **Avg / median % per leg:** 2.66% / 1.93%
- **Sum % (uncompounded):** 74.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.32% | -18.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.32% | -18.5% |
| SELL (all) | 20 | 15 | 75.0% | 6 | 7 | 7 | 4.65% | 93.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 15 | 75.0% | 6 | 7 | 7 | 4.65% | 93.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 15 | 53.6% | 6 | 15 | 7 | 2.66% | 74.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 731.35 | 792.30 | 792.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 720.35 | 790.42 | 791.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 762.30 | 759.50 | 773.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 762.75 | 759.50 | 773.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 773.00 | 759.58 | 772.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 772.55 | 759.58 | 772.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 774.95 | 759.73 | 772.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 796.85 | 759.73 | 772.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 796.95 | 760.10 | 772.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 789.85 | 771.46 | 777.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 788.15 | 771.86 | 777.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 787.55 | 772.02 | 777.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 750.36 | 771.66 | 777.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 748.74 | 771.66 | 777.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 748.17 | 771.66 | 777.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-05 12:15:00 | 710.87 | 764.46 | 772.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 770.25 | 726.41 | 726.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 780.10 | 742.11 | 735.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 742.20 | 744.45 | 736.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 742.20 | 744.45 | 736.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 742.20 | 744.45 | 736.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 738.40 | 744.45 | 736.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 736.60 | 744.37 | 736.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:15:00 | 733.45 | 744.37 | 736.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 734.00 | 744.27 | 736.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 733.40 | 744.27 | 736.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 736.15 | 744.19 | 736.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 735.50 | 744.19 | 736.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 736.60 | 744.03 | 736.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 739.20 | 744.03 | 736.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 738.00 | 743.97 | 736.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:30:00 | 737.00 | 743.97 | 736.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 740.65 | 743.89 | 736.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 737.00 | 743.89 | 736.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 738.55 | 744.11 | 737.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 739.30 | 744.11 | 737.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 742.15 | 744.09 | 737.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 749.40 | 744.06 | 737.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 750.40 | 744.11 | 737.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 751.65 | 744.29 | 737.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:15:00 | 748.00 | 744.10 | 737.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 738.45 | 746.56 | 739.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 738.45 | 746.56 | 739.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 739.80 | 746.49 | 739.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:15:00 | 737.80 | 746.49 | 739.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 737.45 | 746.40 | 739.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 736.95 | 746.31 | 739.92 | SL hit (close<static) qty=1.00 sl=737.05 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 706.60 | 735.92 | 736.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 701.00 | 735.57 | 735.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 722.85 | 721.12 | 727.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 12:15:00 | 727.30 | 721.19 | 727.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 727.30 | 721.19 | 727.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:00:00 | 727.30 | 721.19 | 727.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 728.30 | 721.26 | 727.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:00:00 | 728.30 | 721.26 | 727.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 727.85 | 721.32 | 727.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 731.35 | 721.32 | 727.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 720.00 | 721.31 | 727.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 703.70 | 721.31 | 727.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 09:15:00 | 633.33 | 718.34 | 725.75 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-26 15:00:00 | 789.85 | 2025-09-01 09:15:00 | 750.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 10:15:00 | 788.15 | 2025-09-01 09:15:00 | 748.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 11:00:00 | 787.55 | 2025-09-01 09:15:00 | 748.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 15:00:00 | 789.85 | 2025-09-05 12:15:00 | 710.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-28 10:15:00 | 788.15 | 2025-09-10 09:15:00 | 772.90 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-08-28 11:00:00 | 787.55 | 2025-09-10 09:15:00 | 772.90 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-09-22 09:15:00 | 747.85 | 2025-09-23 09:15:00 | 710.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 747.85 | 2025-09-25 15:15:00 | 673.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-15 12:15:00 | 732.40 | 2025-10-15 14:15:00 | 743.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-16 10:00:00 | 731.40 | 2025-10-31 12:15:00 | 694.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 12:15:00 | 730.45 | 2025-10-31 12:15:00 | 693.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 09:30:00 | 732.30 | 2025-10-31 12:15:00 | 695.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:00:00 | 731.40 | 2025-11-10 09:15:00 | 658.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:15:00 | 730.45 | 2025-11-10 09:15:00 | 657.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-28 09:30:00 | 732.30 | 2025-11-10 09:15:00 | 659.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 719.50 | 2025-11-20 11:15:00 | 723.70 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-21 09:15:00 | 715.65 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-11-21 12:45:00 | 718.95 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-24 09:30:00 | 719.80 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-26 12:15:00 | 749.40 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-12-26 14:15:00 | 750.40 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-12-29 09:30:00 | 751.65 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-30 12:15:00 | 748.00 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-06 09:45:00 | 743.40 | 2026-01-06 10:15:00 | 735.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-06 12:30:00 | 742.50 | 2026-01-06 13:15:00 | 735.55 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-07 11:30:00 | 742.40 | 2026-01-09 09:15:00 | 706.60 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2026-01-07 12:30:00 | 742.90 | 2026-01-09 09:15:00 | 706.60 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 703.70 | 2026-02-05 09:15:00 | 633.33 | TARGET_HIT | 1.00 | 10.00% |
