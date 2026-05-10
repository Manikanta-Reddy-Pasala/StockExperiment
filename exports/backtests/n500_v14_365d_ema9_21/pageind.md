# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 37365.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 21 |
| ALERT3 | 138 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 54 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 31
- **Target hits / Stop hits / Partials:** 5 / 53 / 6
- **Avg / median % per leg:** 1.39% / 0.12%
- **Sum % (uncompounded):** 88.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 0 | 22 | 0 | -0.01% | -0.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.86% | -2.6% |
| BUY @ 3rd Alert (retest2) | 19 | 6 | 31.6% | 0 | 19 | 0 | 0.12% | 2.4% |
| SELL (all) | 42 | 27 | 64.3% | 5 | 31 | 6 | 2.12% | 88.9% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.31% | 0.3% |
| SELL @ 3rd Alert (retest2) | 41 | 26 | 63.4% | 5 | 30 | 6 | 2.16% | 88.6% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.57% | -2.3% |
| retest2 (combined) | 60 | 32 | 53.3% | 5 | 49 | 6 | 1.52% | 90.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 46860.00 | 45691.67 | 45669.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 47020.00 | 46453.52 | 46098.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 46525.00 | 46624.52 | 46338.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 46525.00 | 46624.52 | 46338.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 46375.00 | 46515.73 | 46372.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:45:00 | 47110.00 | 46478.04 | 46403.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 46840.00 | 46641.95 | 46496.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 15:15:00 | 47455.00 | 47642.81 | 47654.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-21 15:15:00 | 47455.00 | 47642.81 | 47654.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 47455.00 | 47642.81 | 47654.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 47430.00 | 47600.25 | 47634.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 47730.00 | 47626.20 | 47642.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 47730.00 | 47626.20 | 47642.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 47920.00 | 47684.96 | 47668.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 47450.00 | 47692.13 | 47711.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 47325.00 | 47601.57 | 47665.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 46575.00 | 46184.84 | 46452.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 46575.00 | 46184.84 | 46452.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 46100.00 | 46167.87 | 46420.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 46805.00 | 46167.87 | 46420.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 46770.00 | 46288.30 | 46452.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 46770.00 | 46288.30 | 46452.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 46495.00 | 46329.64 | 46456.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 46310.00 | 46315.71 | 46438.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 46510.00 | 46085.78 | 46044.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 46510.00 | 46085.78 | 46044.14 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 45755.00 | 46010.81 | 46032.19 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 46100.00 | 46023.26 | 46019.61 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 45950.00 | 46008.60 | 46013.29 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 46250.00 | 46056.88 | 46034.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 46765.00 | 46312.17 | 46178.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 46650.00 | 46663.55 | 46459.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 47125.00 | 46663.55 | 46459.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 12:45:00 | 46815.00 | 46847.26 | 46629.40 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 46745.00 | 46826.81 | 46639.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 46765.00 | 46826.81 | 46639.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 46695.00 | 46780.16 | 46649.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 46760.00 | 46780.16 | 46649.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 46820.00 | 46788.13 | 46665.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 46605.00 | 46751.50 | 46659.86 | SL hit (close<ema400) qty=1.00 sl=46659.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 46605.00 | 46751.50 | 46659.86 | SL hit (close<ema400) qty=1.00 sl=46659.86 alert=retest1 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 46110.00 | 46609.65 | 46615.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 45750.00 | 46340.86 | 46482.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 46195.00 | 46078.41 | 46273.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 46195.00 | 46078.41 | 46273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 46275.00 | 46117.73 | 46273.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 45870.00 | 46117.73 | 46273.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 45850.00 | 46064.18 | 46234.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 45485.00 | 45773.62 | 45897.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:00:00 | 45530.00 | 45724.90 | 45864.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 45440.00 | 45110.19 | 45093.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 45440.00 | 45110.19 | 45093.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 45440.00 | 45110.19 | 45093.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 46495.00 | 45449.52 | 45254.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 45875.00 | 45888.24 | 45562.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 45920.00 | 45888.24 | 45562.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 48490.00 | 48686.64 | 48152.61 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 47390.00 | 48343.62 | 48461.54 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 48395.00 | 48197.99 | 48179.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 48685.00 | 48403.18 | 48291.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 48480.00 | 48626.36 | 48490.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 48480.00 | 48626.36 | 48490.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 48950.00 | 48691.09 | 48532.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 49025.00 | 48725.82 | 48627.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 48495.00 | 48603.89 | 48609.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 48495.00 | 48603.89 | 48609.78 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 48740.00 | 48612.12 | 48608.82 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 48370.00 | 48563.70 | 48587.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 48115.00 | 48389.51 | 48490.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 48180.00 | 48101.26 | 48259.11 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 48835.00 | 48342.45 | 48318.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 48880.00 | 48459.17 | 48376.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 48670.00 | 48814.21 | 48642.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 48670.00 | 48814.21 | 48642.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 48355.00 | 48722.37 | 48616.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 48355.00 | 48722.37 | 48616.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 47975.00 | 48572.89 | 48558.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 47975.00 | 48572.89 | 48558.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 47990.00 | 48456.31 | 48506.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 47440.00 | 48253.05 | 48409.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 46480.00 | 46424.75 | 46645.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:00:00 | 46480.00 | 46424.75 | 46645.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 46600.00 | 46472.64 | 46629.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 46570.00 | 46472.64 | 46629.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 47535.00 | 46685.11 | 46712.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 47535.00 | 46685.11 | 46712.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 47200.00 | 46788.09 | 46756.60 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 46420.00 | 46781.47 | 46808.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 46300.00 | 46574.99 | 46697.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 46655.00 | 46578.99 | 46677.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 46725.00 | 46578.99 | 46677.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 46720.00 | 46607.20 | 46681.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 46710.00 | 46607.20 | 46681.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 46515.00 | 46588.76 | 46666.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:30:00 | 46645.00 | 46588.76 | 46666.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 46460.00 | 46531.84 | 46618.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 46460.00 | 46531.84 | 46618.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 46850.00 | 46598.38 | 46633.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 46850.00 | 46598.38 | 46633.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 46880.00 | 46654.70 | 46656.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 46880.00 | 46654.70 | 46656.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 46995.00 | 46722.76 | 46686.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 47305.00 | 46839.21 | 46743.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 47910.00 | 48380.96 | 48014.23 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 46920.00 | 47658.83 | 47751.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 46755.00 | 47478.06 | 47660.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 46195.00 | 45944.00 | 46300.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 46195.00 | 45944.00 | 46300.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 46165.00 | 46029.96 | 46279.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 46375.00 | 46029.96 | 46279.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 46315.00 | 46086.97 | 46283.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 46315.00 | 46086.97 | 46283.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 46160.00 | 46101.57 | 46271.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 46165.00 | 46101.57 | 46271.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 45860.00 | 46053.26 | 46234.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 44000.00 | 45853.35 | 46040.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 44515.00 | 44111.76 | 44076.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 44515.00 | 44111.76 | 44076.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 44605.00 | 44210.41 | 44124.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 45215.00 | 45275.06 | 44883.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 45215.00 | 45275.06 | 44883.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 45715.00 | 45873.29 | 45584.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 45505.00 | 45873.29 | 45584.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 45615.00 | 45807.73 | 45626.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 45555.00 | 45807.73 | 45626.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 45665.00 | 45779.18 | 45629.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 45665.00 | 45779.18 | 45629.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 45650.00 | 45753.34 | 45631.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 45640.00 | 45753.34 | 45631.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 45585.00 | 45719.68 | 45627.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 46530.00 | 45719.68 | 45627.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 46540.00 | 45883.74 | 45710.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 46710.00 | 46159.59 | 45873.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 46780.00 | 46257.68 | 45943.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 46860.00 | 46644.72 | 46256.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 45715.00 | 46057.84 | 46100.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 44855.00 | 45817.27 | 45987.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 44420.00 | 44356.83 | 44909.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 44500.00 | 44356.83 | 44909.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 44520.00 | 44326.35 | 44676.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 44520.00 | 44326.35 | 44676.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 44685.00 | 44398.08 | 44677.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 44665.00 | 44398.08 | 44677.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 44620.00 | 44442.46 | 44671.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:15:00 | 44825.00 | 44442.46 | 44671.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 44840.00 | 44521.97 | 44687.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 44840.00 | 44521.97 | 44687.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 44830.00 | 44583.58 | 44700.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 44830.00 | 44583.58 | 44700.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 44530.00 | 44643.40 | 44700.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 44495.00 | 44635.72 | 44691.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 44815.00 | 44697.86 | 44712.05 | SL hit (close>static) qty=1.00 sl=44805.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 45245.00 | 44807.29 | 44760.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 45590.00 | 44988.26 | 44852.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 45225.00 | 45251.09 | 45074.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:15:00 | 45515.00 | 45251.09 | 45074.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 45050.00 | 45219.50 | 45091.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 45050.00 | 45219.50 | 45091.27 | SL hit (close<ema400) qty=1.00 sl=45091.27 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 44970.00 | 45219.50 | 45091.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 44890.00 | 45153.60 | 45072.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 44890.00 | 45153.60 | 45072.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 44935.00 | 45109.88 | 45060.43 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 44610.00 | 44971.52 | 45003.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 44585.00 | 44842.58 | 44933.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 44455.00 | 44430.64 | 44597.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 44455.00 | 44430.64 | 44597.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 44270.00 | 44407.21 | 44558.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 44240.00 | 44407.21 | 44558.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 44245.00 | 44256.02 | 44397.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:15:00 | 44245.00 | 44372.74 | 44406.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 44260.00 | 44387.15 | 44406.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 44475.00 | 44404.72 | 44412.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 44475.00 | 44404.72 | 44412.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 44410.00 | 44405.78 | 44412.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 44375.00 | 44405.78 | 44412.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 44215.00 | 44367.62 | 44394.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 44110.00 | 44316.10 | 44368.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:00:00 | 44110.00 | 44256.30 | 44331.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 44500.00 | 44300.49 | 44291.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 45100.00 | 44460.39 | 44365.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 45395.00 | 45397.98 | 45095.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:30:00 | 45355.00 | 45397.98 | 45095.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 44590.00 | 45214.31 | 45063.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 44590.00 | 45214.31 | 45063.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 44390.00 | 45049.44 | 45002.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 44360.00 | 45049.44 | 45002.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 44155.00 | 44870.56 | 44925.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 43930.00 | 44388.49 | 44637.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 42580.00 | 42577.38 | 42894.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:00:00 | 42580.00 | 42577.38 | 42894.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 42770.00 | 42649.12 | 42873.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 42390.00 | 42647.79 | 42822.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 42110.00 | 41496.85 | 41439.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 42110.00 | 41496.85 | 41439.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 42325.00 | 41882.59 | 41660.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 41915.00 | 41927.69 | 41740.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 41915.00 | 41927.69 | 41740.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 42125.00 | 42285.53 | 42141.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 42115.00 | 42285.53 | 42141.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 42100.00 | 42248.43 | 42137.63 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 41665.00 | 42095.44 | 42108.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 41625.00 | 41850.46 | 41950.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 41860.00 | 41850.69 | 41932.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 41900.00 | 41850.69 | 41932.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 41715.00 | 41711.03 | 41829.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 41800.00 | 41711.03 | 41829.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 41790.00 | 41725.06 | 41815.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 41690.00 | 41733.64 | 41803.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 40975.00 | 40840.11 | 40834.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 40975.00 | 40840.11 | 40834.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 41190.00 | 41018.92 | 40949.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 41460.00 | 41671.74 | 41496.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 41460.00 | 41671.74 | 41496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 41460.00 | 41629.39 | 41492.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 41385.00 | 41629.39 | 41492.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 41500.00 | 41603.51 | 41493.56 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 41135.00 | 41412.48 | 41424.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 40900.00 | 41309.98 | 41377.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 40905.00 | 40867.35 | 41011.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 40905.00 | 40867.35 | 41011.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 41000.00 | 40893.88 | 41010.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 41190.00 | 40893.88 | 41010.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 40920.00 | 40899.10 | 41002.34 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 41190.00 | 41061.06 | 41054.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 41335.00 | 41115.85 | 41080.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 40790.00 | 41085.74 | 41074.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 40790.00 | 41085.74 | 41074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 40925.00 | 41053.59 | 41061.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 40680.00 | 40938.30 | 41004.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 41000.00 | 40887.77 | 40959.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 40980.00 | 40887.77 | 40959.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 41020.00 | 40914.22 | 40964.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 41125.00 | 40914.22 | 40964.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 41030.00 | 40937.37 | 40970.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 41015.00 | 40937.37 | 40970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 41185.00 | 40986.90 | 40990.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 41185.00 | 40986.90 | 40990.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 41160.00 | 41021.52 | 41005.68 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 40825.00 | 40995.38 | 41003.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 15:15:00 | 40600.00 | 40785.05 | 40879.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 39735.00 | 39734.24 | 39965.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 39700.00 | 39734.24 | 39965.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 39840.00 | 39757.91 | 39936.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 39900.00 | 39757.91 | 39936.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 40020.00 | 39810.33 | 39943.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 40020.00 | 39810.33 | 39943.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 40205.00 | 39889.26 | 39967.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 40205.00 | 39889.26 | 39967.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 40160.00 | 39943.41 | 39985.13 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 40150.00 | 40016.18 | 40012.91 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 39835.00 | 39986.96 | 40000.63 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 40085.00 | 40008.65 | 40008.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 40240.00 | 40059.63 | 40032.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 39720.00 | 40646.97 | 40517.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:45:00 | 39935.00 | 40646.97 | 40517.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 39350.00 | 40387.57 | 40411.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 38990.00 | 39549.18 | 39816.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 15:15:00 | 39375.00 | 39336.56 | 39563.32 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:15:00 | 39030.00 | 39336.56 | 39563.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 38910.00 | 38778.62 | 38882.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 38910.00 | 38778.62 | 38882.09 | SL hit (close>ema400) qty=1.00 sl=38882.09 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 38910.00 | 38778.62 | 38882.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 38755.00 | 38773.90 | 38870.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 39120.00 | 38773.90 | 38870.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 39020.00 | 38823.12 | 38884.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 38990.00 | 38823.12 | 38884.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 38925.00 | 38843.49 | 38887.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 38800.00 | 38835.79 | 38880.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:30:00 | 38850.00 | 38834.71 | 38872.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 38835.00 | 38846.77 | 38874.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 38865.00 | 38710.33 | 38751.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 38880.00 | 38749.01 | 38762.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 38880.00 | 38749.01 | 38762.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 38920.00 | 38783.21 | 38776.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 39005.00 | 38827.57 | 38797.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 38800.00 | 38877.27 | 38845.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 38800.00 | 38877.27 | 38845.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 38975.00 | 38896.82 | 38857.18 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 38600.00 | 38823.56 | 38829.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 38410.00 | 38740.85 | 38791.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 12:15:00 | 37550.00 | 37519.28 | 37863.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 37550.00 | 37519.28 | 37863.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 37505.00 | 37289.86 | 37503.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 37150.00 | 37353.74 | 37425.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 36840.00 | 37222.27 | 37318.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 37115.00 | 37163.45 | 37271.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:15:00 | 37130.00 | 37169.76 | 37264.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 37280.00 | 37191.81 | 37266.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 37280.00 | 37191.81 | 37266.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 37265.00 | 37206.45 | 37265.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 37180.00 | 37206.45 | 37265.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 37080.00 | 37199.02 | 37244.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:15:00 | 35321.00 | 35703.87 | 35965.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 35730.00 | 35709.10 | 35944.36 | SL hit (close>ema200) qty=0.50 sl=35709.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 36565.00 | 36031.06 | 36012.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 36700.00 | 36315.58 | 36159.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 36375.00 | 36460.90 | 36293.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 36375.00 | 36460.90 | 36293.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 36455.00 | 36459.72 | 36308.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 36350.00 | 36459.72 | 36308.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 36625.00 | 36575.65 | 36443.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 36510.00 | 36575.65 | 36443.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 36360.00 | 36566.16 | 36505.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 36360.00 | 36566.16 | 36505.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 36500.00 | 36552.93 | 36505.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 36535.00 | 36532.34 | 36500.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 36265.00 | 36442.22 | 36464.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 36265.00 | 36442.22 | 36464.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 36115.00 | 36376.78 | 36432.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 35900.00 | 35865.63 | 36084.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 35775.00 | 35865.63 | 36084.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 35655.00 | 35823.51 | 36045.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 35585.00 | 35850.84 | 35962.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 35535.00 | 35850.84 | 35962.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 35625.00 | 35764.94 | 35901.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 35570.00 | 35732.95 | 35874.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 35820.00 | 35709.07 | 35803.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 35860.00 | 35709.07 | 35803.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 35990.00 | 35765.26 | 35820.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 35955.00 | 35765.26 | 35820.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 35830.00 | 35798.96 | 35827.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 35635.00 | 35783.14 | 35814.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33805.75 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33758.25 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33843.75 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33791.50 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 33853.25 | 34149.31 | 34266.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 32026.50 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 31981.50 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 32062.50 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 32013.00 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 32071.50 | 33511.58 | 33861.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 33015.00 | 32628.19 | 32601.38 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 32685.00 | 32817.66 | 32818.54 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 33000.00 | 32820.47 | 32813.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 33605.00 | 32977.38 | 32885.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 09:15:00 | 35195.00 | 35295.72 | 34828.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 35195.00 | 35295.72 | 34828.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 34725.00 | 35126.26 | 34829.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 34725.00 | 35126.26 | 34829.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 34640.00 | 35029.01 | 34812.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 34490.00 | 35029.01 | 34812.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 34815.00 | 34958.37 | 34815.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:15:00 | 34850.00 | 34958.37 | 34815.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 34850.00 | 34936.69 | 34818.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 34920.00 | 34936.69 | 34818.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 34745.00 | 34898.35 | 34812.23 | SL hit (close<static) qty=1.00 sl=34780.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 34580.00 | 34733.31 | 34752.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 34305.00 | 34612.77 | 34689.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 13:15:00 | 34550.00 | 34453.12 | 34574.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 34550.00 | 34453.12 | 34574.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 34390.00 | 34440.50 | 34557.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 33925.00 | 34437.40 | 34545.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 33120.00 | 33015.57 | 33014.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 33120.00 | 33015.57 | 33014.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 33360.00 | 33097.25 | 33052.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 33085.00 | 33106.44 | 33065.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:00:00 | 33085.00 | 33106.44 | 33065.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 33030.00 | 33091.15 | 33061.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 33030.00 | 33091.15 | 33061.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 32870.00 | 33046.92 | 33044.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 32870.00 | 33046.92 | 33044.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 32725.00 | 32982.54 | 33015.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 15:15:00 | 32670.00 | 32883.62 | 32962.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 32095.00 | 32094.25 | 32350.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:45:00 | 32185.00 | 32094.25 | 32350.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 31270.00 | 31083.08 | 31361.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 31270.00 | 31083.08 | 31361.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 31490.00 | 31164.47 | 31373.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 31490.00 | 31164.47 | 31373.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 31580.00 | 31247.57 | 31392.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:30:00 | 31580.00 | 31247.57 | 31392.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 31440.00 | 31329.00 | 31398.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 31695.00 | 31329.00 | 31398.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 31485.00 | 31360.20 | 31406.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 31340.00 | 31390.16 | 31415.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 31695.00 | 31481.00 | 31453.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 31695.00 | 31481.00 | 31453.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 31725.00 | 31529.80 | 31477.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 31120.00 | 31427.15 | 31448.28 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 31610.00 | 31331.33 | 31321.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 31860.00 | 31466.45 | 31386.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 31480.00 | 31549.46 | 31460.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 31480.00 | 31549.46 | 31460.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 31745.00 | 31588.57 | 31486.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 31080.00 | 31588.57 | 31486.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 31015.00 | 31473.85 | 31443.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 31040.00 | 31473.85 | 31443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 31040.00 | 31387.08 | 31407.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 12:15:00 | 30790.00 | 31211.33 | 31320.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 30220.00 | 30150.44 | 30437.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 30255.00 | 30150.44 | 30437.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 30445.00 | 30193.28 | 30406.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 30380.00 | 30193.28 | 30406.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 30470.00 | 30248.63 | 30412.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 30530.00 | 30248.63 | 30412.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 30515.00 | 30301.90 | 30421.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 30515.00 | 30301.90 | 30421.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 30615.00 | 30364.52 | 30439.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 30615.00 | 30364.52 | 30439.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 30645.00 | 30420.62 | 30457.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 30645.00 | 30420.62 | 30457.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 31190.00 | 30603.92 | 30534.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 31335.00 | 30750.13 | 30607.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 31265.00 | 31359.19 | 31036.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:30:00 | 31145.00 | 31359.19 | 31036.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 31335.00 | 31328.08 | 31076.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 31370.00 | 31335.47 | 31103.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:45:00 | 31355.00 | 31329.37 | 31121.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 31920.00 | 31279.20 | 31133.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:00:00 | 31470.00 | 31746.93 | 31562.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 31395.00 | 31633.83 | 31540.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 31395.00 | 31633.83 | 31540.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 31375.00 | 31571.85 | 31527.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 31375.00 | 31571.85 | 31527.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 31990.00 | 31576.71 | 31531.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 32125.00 | 31686.37 | 31585.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 32675.00 | 32746.57 | 32313.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 32675.00 | 32746.57 | 32313.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 32025.00 | 32578.81 | 32310.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 32025.00 | 32578.81 | 32310.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 32120.00 | 32487.04 | 32293.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 32075.00 | 32487.04 | 32293.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 32140.00 | 32381.31 | 32276.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 32140.00 | 32381.31 | 32276.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 31975.00 | 32300.05 | 32249.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 31975.00 | 32300.05 | 32249.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 31800.00 | 32200.04 | 32208.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 31525.00 | 31828.67 | 31994.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 32535.00 | 31880.95 | 31988.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 32430.00 | 31990.76 | 32028.61 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 32445.00 | 32081.61 | 32066.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 32730.00 | 32211.29 | 32126.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 35760.00 | 35789.37 | 35372.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 09:15:00 | 35945.00 | 35789.37 | 35372.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 36090.00 | 36092.79 | 35796.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 36230.00 | 36115.23 | 35834.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 36225.00 | 36136.19 | 35869.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 36435.00 | 36048.24 | 35911.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 36240.00 | 36067.59 | 35933.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 37940.00 | 38054.94 | 37904.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 37980.00 | 38054.94 | 37904.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 37810.00 | 37978.77 | 37904.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 37810.00 | 37978.77 | 37904.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 37765.00 | 37936.01 | 37892.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 37765.00 | 37936.01 | 37892.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 37825.00 | 37895.25 | 37880.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 37700.00 | 37895.25 | 37880.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 37725.00 | 37861.20 | 37865.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 37415.00 | 37771.96 | 37824.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 37755.00 | 37679.36 | 37761.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 37755.00 | 37679.36 | 37761.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 37960.00 | 37735.49 | 37779.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 37960.00 | 37735.49 | 37779.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 38000.00 | 37788.39 | 37799.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 38070.00 | 37788.39 | 37799.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 38130.00 | 37856.71 | 37829.86 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 37665.00 | 37798.84 | 37809.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 10:15:00 | 37575.00 | 37689.56 | 37744.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 37770.00 | 37680.12 | 37729.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 37765.00 | 37680.12 | 37729.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 37685.00 | 37681.10 | 37725.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 37600.00 | 37685.88 | 37723.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 37635.00 | 37674.96 | 37710.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 37535.00 | 37636.97 | 37690.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 37450.00 | 37509.37 | 37603.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 37515.00 | 37508.99 | 37587.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 37305.00 | 37455.20 | 37555.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 37405.00 | 37048.76 | 37027.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 37625.00 | 37271.47 | 37149.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 37370.00 | 37409.87 | 37276.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 37370.00 | 37409.87 | 37276.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 37285.00 | 37384.89 | 37277.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 37285.00 | 37384.89 | 37277.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 37285.00 | 37364.92 | 37278.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:00:00 | 37335.00 | 37358.93 | 37283.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:30:00 | 37335.00 | 37350.15 | 37286.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 37415.00 | 37324.69 | 37285.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 37120.00 | 37283.75 | 37270.33 | SL hit (close<static) qty=1.00 sl=37180.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 37120.00 | 37283.75 | 37270.33 | SL hit (close<static) qty=1.00 sl=37180.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 37120.00 | 37283.75 | 37270.33 | SL hit (close<static) qty=1.00 sl=37180.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 37000.00 | 37227.00 | 37245.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 36920.00 | 37128.48 | 37195.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 37235.00 | 37149.79 | 37199.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 37235.00 | 37149.79 | 37199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 37335.00 | 37186.83 | 37211.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 37470.00 | 37186.83 | 37211.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 14:45:00 | 47110.00 | 2025-05-21 15:15:00 | 47455.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-05-16 10:00:00 | 46840.00 | 2025-05-21 15:15:00 | 47455.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2025-05-30 11:45:00 | 46310.00 | 2025-06-05 09:15:00 | 46510.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-11 09:15:00 | 47125.00 | 2025-06-12 10:15:00 | 46605.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-06-11 12:45:00 | 46815.00 | 2025-06-12 10:15:00 | 46605.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-18 09:15:00 | 45485.00 | 2025-06-23 14:15:00 | 45440.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-06-18 10:00:00 | 45530.00 | 2025-06-23 14:15:00 | 45440.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-08 15:15:00 | 49025.00 | 2025-07-09 13:15:00 | 48495.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-08 09:15:00 | 44000.00 | 2025-08-18 12:15:00 | 44515.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-25 11:30:00 | 46710.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-25 13:15:00 | 46780.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-26 09:45:00 | 46860.00 | 2025-08-26 15:15:00 | 45715.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-02 11:30:00 | 44495.00 | 2025-09-02 13:15:00 | 44815.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-09-04 09:15:00 | 45515.00 | 2025-09-04 10:15:00 | 45050.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-08 15:15:00 | 44240.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-09 14:15:00 | 44245.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-10 12:15:00 | 44245.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-10 13:30:00 | 44260.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-11 11:00:00 | 44110.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-11 13:00:00 | 44110.00 | 2025-09-15 15:15:00 | 44500.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-25 15:00:00 | 42390.00 | 2025-10-01 12:15:00 | 42110.00 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-10-10 12:15:00 | 41690.00 | 2025-10-16 13:15:00 | 40975.00 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest1 | 2025-11-19 09:15:00 | 39030.00 | 2025-11-21 14:15:00 | 38910.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-11-24 11:45:00 | 38800.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-24 13:30:00 | 38850.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-11-24 15:15:00 | 38835.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-26 11:15:00 | 38865.00 | 2025-11-26 13:15:00 | 38920.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-12-08 10:30:00 | 37150.00 | 2025-12-19 11:15:00 | 35321.00 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-08 10:30:00 | 37150.00 | 2025-12-19 12:15:00 | 35730.00 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-12-09 09:15:00 | 36840.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-12-09 11:15:00 | 37115.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-12-09 12:15:00 | 37130.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-12-09 14:15:00 | 37180.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-10 10:45:00 | 37080.00 | 2025-12-22 10:15:00 | 36565.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-12-26 11:45:00 | 36535.00 | 2025-12-29 09:15:00 | 36265.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-01 09:30:00 | 35585.00 | 2026-01-20 09:15:00 | 33805.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 35535.00 | 2026-01-20 09:15:00 | 33758.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 11:45:00 | 35625.00 | 2026-01-20 09:15:00 | 33843.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 12:30:00 | 35570.00 | 2026-01-20 09:15:00 | 33791.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 35635.00 | 2026-01-20 09:15:00 | 33853.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 35585.00 | 2026-01-20 15:15:00 | 32026.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 35535.00 | 2026-01-20 15:15:00 | 31981.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 11:45:00 | 35625.00 | 2026-01-20 15:15:00 | 32062.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 12:30:00 | 35570.00 | 2026-01-20 15:15:00 | 32013.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 35635.00 | 2026-01-20 15:15:00 | 32071.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-10 09:15:00 | 34920.00 | 2026-02-10 09:15:00 | 34745.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-12 09:15:00 | 33925.00 | 2026-02-24 12:15:00 | 33120.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2026-03-06 09:30:00 | 31340.00 | 2026-03-06 12:15:00 | 31695.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-19 12:45:00 | 31370.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-03-19 13:45:00 | 31355.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-03-20 09:15:00 | 31920.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-23 11:00:00 | 31470.00 | 2026-03-23 15:15:00 | 31175.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-13 11:15:00 | 36230.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.13% |
| BUY | retest2 | 2026-04-13 11:45:00 | 36225.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2026-04-15 09:15:00 | 36435.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 3.54% |
| BUY | retest2 | 2026-04-15 10:15:00 | 36240.00 | 2026-04-23 09:15:00 | 37725.00 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-27 15:15:00 | 37600.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-28 09:45:00 | 37635.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2026-04-28 10:30:00 | 37535.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2026-04-28 15:00:00 | 37450.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-04-29 10:30:00 | 37305.00 | 2026-05-05 14:15:00 | 37405.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-05-07 13:00:00 | 37335.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-05-07 13:30:00 | 37335.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-05-08 09:15:00 | 37415.00 | 2026-05-08 09:15:00 | 37120.00 | STOP_HIT | 1.00 | -0.79% |
