# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 955.20
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -2.49% / -1.79%
- **Sum % (uncompounded):** -27.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.49% | -27.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.49% | -27.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.49% | -27.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 631.00 | 534.45 | 533.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 634.00 | 536.33 | 534.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 831.40 | 832.68 | 763.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:45:00 | 833.20 | 832.68 | 763.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 772.75 | 823.70 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 773.15 | 823.70 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 771.70 | 823.18 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 771.95 | 823.18 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 772.00 | 822.67 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 772.90 | 822.67 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 769.15 | 822.14 | 772.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 769.15 | 822.14 | 772.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 770.50 | 821.63 | 772.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:15:00 | 766.50 | 821.63 | 772.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 766.50 | 821.08 | 771.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 777.10 | 821.08 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 779.10 | 820.66 | 772.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 786.45 | 810.28 | 770.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 788.00 | 810.28 | 770.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:30:00 | 785.00 | 809.56 | 771.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 806.50 | 808.44 | 771.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 776.75 | 806.90 | 773.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 771.55 | 806.90 | 773.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 774.00 | 806.58 | 773.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 778.20 | 806.58 | 773.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 758.05 | 806.09 | 773.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 753.00 | 805.56 | 772.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 753.15 | 805.56 | 772.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 764.00 | 787.90 | 768.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 771.35 | 787.90 | 768.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 772.80 | 787.78 | 768.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:00:00 | 779.35 | 787.48 | 768.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 771.85 | 786.57 | 768.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 769.35 | 786.46 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 769.35 | 786.46 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 768.05 | 786.27 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 768.05 | 786.27 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 769.80 | 786.11 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 769.80 | 786.11 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 769.00 | 785.94 | 769.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 776.10 | 785.59 | 769.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 771.20 | 785.26 | 769.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 773.35 | 781.68 | 769.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 765.00 | 780.97 | 769.64 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-09-03 11:15:00 | 721.00 | 760.16 | 760.33 | min_gap filter: gap=0.023% < 0.030% |
| TREND_RESET | 2025-09-03 11:15:00 | 721.00 | 760.16 | 760.33 | EMA inversion without crossover edge (EMA200=760.16 EMA400=760.33) — end cycle |
| CROSSOVER_SKIP | 2025-09-12 14:15:00 | 782.10 | 759.39 | 759.35 | min_gap filter: gap=0.004% < 0.030% |
| CROSSOVER_SKIP | 2026-01-01 10:15:00 | 784.45 | 807.04 | 807.13 | min_gap filter: gap=0.012% < 0.030% |

### Cycle 2 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 804.25 | 624.56 | 623.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 815.25 | 626.46 | 624.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-30 14:30:00 | 786.45 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-07-30 15:15:00 | 788.00 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-07-31 10:30:00 | 785.00 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-01 09:15:00 | 806.50 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -6.02% |
| BUY | retest2 | 2025-08-12 09:15:00 | 771.35 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-12 09:45:00 | 772.80 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-12 12:00:00 | 779.35 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-13 09:15:00 | 771.85 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-18 09:15:00 | 776.10 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-18 12:30:00 | 771.20 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-25 10:00:00 | 773.35 | 2025-08-25 14:15:00 | 765.00 | STOP_HIT | 1.00 | -1.08% |
