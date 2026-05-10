# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 525.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 6
- **Target hits / Stop hits / Partials:** 0 / 17 / 11
- **Avg / median % per leg:** 1.91% / 1.72%
- **Sum % (uncompounded):** 53.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.76% | -11.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.76% | -11.0% |
| SELL (all) | 24 | 22 | 91.7% | 0 | 13 | 11 | 2.69% | 64.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 22 | 91.7% | 0 | 13 | 11 | 2.69% | 64.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 22 | 78.6% | 0 | 17 | 11 | 1.91% | 53.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 783.80 | 727.87 | 727.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 861.95 | 730.37 | 729.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 832.00 | 839.88 | 815.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 832.00 | 839.88 | 815.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 817.40 | 839.08 | 816.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 819.95 | 839.08 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 816.00 | 838.85 | 816.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 811.00 | 838.85 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 814.20 | 838.60 | 816.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 823.75 | 835.13 | 815.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:45:00 | 821.05 | 834.40 | 816.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 795.45 | 833.22 | 817.89 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 795.45 | 833.22 | 817.89 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:00:00 | 825.20 | 831.60 | 817.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 802.10 | 830.69 | 817.68 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 820.00 | 810.20 | 809.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 810.25 | 810.30 | 809.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 810.25 | 810.30 | 809.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 806.20 | 810.26 | 809.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-12 13:15:00 | 806.20 | 810.26 | 809.96 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-12 13:45:00 | 806.25 | 810.26 | 809.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 807.30 | 810.23 | 809.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:45:00 | 803.85 | 810.23 | 809.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 796.55 | 810.08 | 809.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 796.55 | 810.08 | 809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 796.25 | 809.94 | 809.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 796.25 | 809.94 | 809.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 800.00 | 809.62 | 809.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 796.20 | 808.65 | 809.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 814.65 | 807.65 | 808.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 815.75 | 807.73 | 808.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 810.50 | 807.73 | 808.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 814.95 | 808.05 | 808.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 814.50 | 808.05 | 808.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 810.85 | 808.22 | 808.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 807.00 | 808.22 | 808.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 802.00 | 807.85 | 808.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:45:00 | 808.05 | 807.86 | 808.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 807.10 | 807.80 | 808.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 13:15:00 | 767.65 | 804.62 | 806.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 13:15:00 | 766.75 | 804.62 | 806.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 766.65 | 803.85 | 806.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 761.90 | 803.85 | 806.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | SL hit (close>ema200) qty=0.50 sl=794.28 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | SL hit (close>ema200) qty=0.50 sl=794.28 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | SL hit (close>ema200) qty=0.50 sl=794.28 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | SL hit (close>ema200) qty=0.50 sl=794.28 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 801.95 | 794.28 | 800.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 818.90 | 794.52 | 800.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 814.45 | 794.52 | 800.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 853.95 | 805.34 | 805.33 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 759.20 | 806.04 | 806.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 13:15:00 | 757.00 | 805.55 | 805.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 793.30 | 789.54 | 796.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:30:00 | 791.45 | 789.54 | 796.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 792.15 | 789.57 | 796.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 790.20 | 789.57 | 796.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 790.50 | 789.59 | 796.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 800.95 | 789.76 | 796.59 | SL hit (close>static) qty=1.00 sl=797.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 800.95 | 789.76 | 796.59 | SL hit (close>static) qty=1.00 sl=797.55 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 789.40 | 789.80 | 796.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:15:00 | 749.93 | 786.30 | 794.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 784.05 | 783.55 | 792.24 | SL hit (close>ema200) qty=0.50 sl=783.55 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 757.15 | 791.04 | 794.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 719.29 | 783.69 | 790.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 743.80 | 742.49 | 761.44 | SL hit (close>ema200) qty=0.50 sl=742.49 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 748.80 | 742.81 | 758.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:00:00 | 747.25 | 742.92 | 758.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:30:00 | 745.60 | 743.01 | 757.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:15:00 | 709.89 | 739.34 | 754.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 740.25 | 736.24 | 749.73 | SL hit (close>ema200) qty=0.50 sl=736.24 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 746.00 | 736.51 | 749.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 747.25 | 736.61 | 749.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 749.10 | 736.73 | 749.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:45:00 | 748.25 | 736.73 | 749.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 751.05 | 736.87 | 749.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 751.05 | 736.87 | 749.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 750.20 | 737.01 | 749.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:00:00 | 745.35 | 737.09 | 749.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 708.32 | 735.12 | 746.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 708.70 | 735.12 | 746.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 709.89 | 735.12 | 746.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 708.08 | 735.12 | 746.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 733.15 | 722.00 | 736.27 | SL hit (close>ema200) qty=0.50 sl=722.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 733.15 | 722.00 | 736.27 | SL hit (close>ema200) qty=0.50 sl=722.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 733.15 | 722.00 | 736.27 | SL hit (close>ema200) qty=0.50 sl=722.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 733.15 | 722.00 | 736.27 | SL hit (close>ema200) qty=0.50 sl=722.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-16 09:15:00 | 823.75 | 2025-07-23 09:15:00 | 795.45 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-18 10:45:00 | 821.05 | 2025-07-23 09:15:00 | 795.45 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-07-24 10:00:00 | 825.20 | 2025-07-25 10:15:00 | 802.10 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-08-12 10:00:00 | 820.00 | 2025-08-12 13:15:00 | 806.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-08-21 15:15:00 | 807.00 | 2025-08-29 13:15:00 | 767.65 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-08-26 09:15:00 | 802.00 | 2025-08-29 13:15:00 | 766.75 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-08-26 11:45:00 | 808.05 | 2025-08-29 15:15:00 | 766.65 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-08-26 15:00:00 | 807.10 | 2025-08-29 15:15:00 | 761.90 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-08-21 15:15:00 | 807.00 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-08-26 09:15:00 | 802.00 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.01% |
| SELL | retest2 | 2025-08-26 11:45:00 | 808.05 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2025-08-26 15:00:00 | 807.10 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2025-10-10 11:15:00 | 790.20 | 2025-10-10 14:15:00 | 800.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-10 12:15:00 | 790.50 | 2025-10-10 14:15:00 | 800.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-10-13 09:15:00 | 789.40 | 2025-10-15 11:15:00 | 749.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 789.40 | 2025-10-17 11:15:00 | 784.05 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2025-11-03 09:15:00 | 757.15 | 2025-11-06 11:15:00 | 719.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 09:15:00 | 757.15 | 2025-11-27 11:15:00 | 743.80 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-12-05 12:00:00 | 747.25 | 2025-12-11 09:15:00 | 709.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 12:00:00 | 747.25 | 2025-12-19 10:15:00 | 740.25 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2025-12-05 13:30:00 | 745.60 | 2025-12-30 09:15:00 | 708.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 09:45:00 | 746.00 | 2025-12-30 09:15:00 | 708.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 10:30:00 | 747.25 | 2025-12-30 09:15:00 | 709.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 15:00:00 | 745.35 | 2025-12-30 09:15:00 | 708.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:30:00 | 745.60 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2025-12-22 09:45:00 | 746.00 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-12-22 10:30:00 | 747.25 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-12-22 15:00:00 | 745.35 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.64% |
