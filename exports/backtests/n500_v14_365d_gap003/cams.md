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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** -1.88% / -1.90%
- **Sum % (uncompounded):** -22.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 1 | 8.3% | 0 | 11 | 1 | -1.88% | -22.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 0 | 11 | 1 | -1.88% | -22.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 1 | 8.3% | 0 | 11 | 1 | -1.88% | -22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 09:15:00)

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
| CROSSOVER_SKIP | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | min_gap filter: gap=0.006% < 0.030% |
| TREND_RESET | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | EMA inversion without crossover edge (EMA200=778.07 EMA400=778.03) — end cycle |
| CROSSOVER_SKIP | 2025-12-08 12:15:00 | 751.50 | 778.44 | 778.56 | min_gap filter: gap=0.015% < 0.030% |

### Cycle 2 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 774.20 | 699.61 | 699.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 779.45 | 720.36 | 711.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
