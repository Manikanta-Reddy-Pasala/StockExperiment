# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 565.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 12
- **Target hits / Stop hits / Partials:** 3 / 18 / 5
- **Avg / median % per leg:** 1.97% / 0.21%
- **Sum % (uncompounded):** 51.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 4 | 25.0% | 0 | 16 | 0 | -0.97% | -15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 0 | 16 | 0 | -0.97% | -15.5% |
| SELL (all) | 10 | 10 | 100.0% | 3 | 2 | 5 | 6.67% | 66.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 3 | 2 | 5 | 6.67% | 66.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 14 | 53.8% | 3 | 18 | 5 | 1.97% | 51.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 13:15:00 | 764.25 | 744.47 | 744.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 773.05 | 745.13 | 744.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 764.70 | 769.22 | 759.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 764.70 | 769.22 | 759.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 761.30 | 769.14 | 759.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 761.30 | 769.14 | 759.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 758.75 | 769.04 | 759.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 758.75 | 769.04 | 759.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 758.45 | 768.93 | 759.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 757.90 | 768.93 | 759.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 757.95 | 768.82 | 759.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 762.00 | 768.23 | 759.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:00:00 | 762.00 | 773.72 | 765.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 762.10 | 773.50 | 765.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 763.50 | 773.50 | 765.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 768.25 | 773.37 | 765.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:30:00 | 770.70 | 773.34 | 765.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:30:00 | 771.00 | 773.24 | 765.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 770.00 | 773.15 | 765.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 769.15 | 773.05 | 765.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 770.10 | 773.02 | 765.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 763.70 | 772.86 | 765.44 | SL hit (close<static) qty=1.00 sl=764.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 729.15 | 766.97 | 767.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 724.75 | 763.29 | 765.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 727.95 | 722.97 | 734.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:45:00 | 729.70 | 722.97 | 734.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 732.35 | 723.68 | 734.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:30:00 | 735.10 | 723.68 | 734.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 733.50 | 723.92 | 734.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 735.50 | 723.92 | 734.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 733.70 | 724.02 | 734.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 734.75 | 724.02 | 734.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 733.35 | 724.18 | 734.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 734.80 | 724.18 | 734.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 736.05 | 724.30 | 734.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 735.25 | 724.30 | 734.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 735.35 | 724.41 | 734.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 735.35 | 724.41 | 734.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 735.30 | 724.72 | 734.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 732.15 | 725.47 | 734.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 732.55 | 725.67 | 734.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:30:00 | 732.35 | 717.64 | 723.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 730.60 | 717.64 | 723.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 717.30 | 719.21 | 723.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:30:00 | 723.20 | 719.21 | 723.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 723.80 | 719.30 | 723.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 723.80 | 719.30 | 723.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 723.50 | 719.34 | 723.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 724.70 | 719.34 | 723.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 724.50 | 719.39 | 723.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 722.75 | 719.39 | 723.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 722.35 | 719.42 | 723.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 720.70 | 719.43 | 723.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 695.54 | 712.78 | 718.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 695.92 | 712.78 | 718.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 695.73 | 712.78 | 718.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:15:00 | 694.07 | 712.59 | 717.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:15:00 | 684.66 | 710.66 | 716.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-09 09:15:00 | 658.93 | 693.95 | 705.17 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-02 12:45:00 | 762.00 | 2025-06-18 11:15:00 | 763.70 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-06-16 10:00:00 | 762.00 | 2025-06-18 11:15:00 | 763.70 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-06-16 11:30:00 | 762.10 | 2025-06-18 11:15:00 | 763.70 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-06-16 12:00:00 | 763.50 | 2025-06-18 11:15:00 | 763.70 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-06-16 14:30:00 | 770.70 | 2025-06-19 09:15:00 | 757.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-17 09:30:00 | 771.00 | 2025-06-19 09:15:00 | 757.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-06-17 12:00:00 | 770.00 | 2025-06-19 09:15:00 | 757.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-18 09:15:00 | 769.15 | 2025-06-19 09:15:00 | 757.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-26 14:30:00 | 772.45 | 2025-07-18 09:15:00 | 768.20 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-27 09:15:00 | 778.40 | 2025-07-18 09:15:00 | 768.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-02 14:00:00 | 773.70 | 2025-07-18 09:15:00 | 768.20 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-15 09:15:00 | 774.80 | 2025-07-18 09:15:00 | 768.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-15 12:15:00 | 773.80 | 2025-07-22 11:15:00 | 762.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-15 15:00:00 | 774.30 | 2025-07-22 11:15:00 | 762.65 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-16 10:15:00 | 773.50 | 2025-07-22 11:15:00 | 762.65 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-16 11:45:00 | 775.00 | 2025-07-22 11:15:00 | 762.65 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-22 11:15:00 | 732.15 | 2025-11-21 09:15:00 | 695.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 732.55 | 2025-11-21 09:15:00 | 695.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 11:30:00 | 732.35 | 2025-11-21 09:15:00 | 695.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 12:15:00 | 730.60 | 2025-11-21 10:15:00 | 694.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:30:00 | 720.70 | 2025-11-24 12:15:00 | 684.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:15:00 | 732.15 | 2025-12-09 09:15:00 | 658.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 732.55 | 2025-12-09 09:15:00 | 659.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-29 11:30:00 | 732.35 | 2025-12-09 09:15:00 | 659.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-29 12:15:00 | 730.60 | 2025-12-23 09:15:00 | 683.00 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2025-11-04 10:30:00 | 720.70 | 2025-12-23 09:15:00 | 683.00 | STOP_HIT | 0.50 | 5.23% |
