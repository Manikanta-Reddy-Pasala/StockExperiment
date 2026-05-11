# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2026-01-19 09:15:00 → 2026-05-08 15:15:00 (518 bars)
- **Last close:** 37365.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 15 |
| ALERT2 | 15 |
| ALERT2_SKIP | 5 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 10
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** 0.61% / -0.27%
- **Sum % (uncompounded):** 11.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 0 | 12 | 0 | 0.75% | 9.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 0 | 12 | 0 | 0.75% | 9.0% |
| SELL (all) | 7 | 5 | 71.4% | 0 | 7 | 0 | 0.37% | 2.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 5 | 71.4% | 0 | 7 | 0 | 0.37% | 2.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 9 | 47.4% | 0 | 19 | 0 | 0.61% | 11.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 33015.00 | 32628.19 | 32604.79 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 32685.00 | 32817.66 | 32819.22 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 33000.00 | 32820.47 | 32814.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 33605.00 | 32977.38 | 32885.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 09:15:00 | 35195.00 | 35295.72 | 34828.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 35195.00 | 35295.72 | 34828.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 34725.00 | 35126.26 | 34829.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 34725.00 | 35126.26 | 34829.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 34640.00 | 35029.01 | 34812.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 34490.00 | 35029.01 | 34812.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 34815.00 | 34958.37 | 34815.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:15:00 | 34850.00 | 34958.37 | 34815.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 34850.00 | 34936.69 | 34818.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 34920.00 | 34936.69 | 34818.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 34745.00 | 34898.35 | 34812.25 | SL hit (close<static) qty=1.00 sl=34780.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 34580.00 | 34733.31 | 34752.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 34305.00 | 34612.77 | 34689.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 13:15:00 | 34550.00 | 34453.12 | 34574.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 34550.00 | 34453.12 | 34574.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 34390.00 | 34440.50 | 34557.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 33925.00 | 34437.40 | 34545.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 33120.00 | 33015.57 | 33014.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-24 12:15:00)

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

### Cycle 6 — SELL (started 2026-02-25 13:15:00)

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

### Cycle 7 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 31695.00 | 31481.00 | 31453.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 31725.00 | 31529.80 | 31477.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 31285.00 | 31503.94 | 31481.11 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 31120.00 | 31427.15 | 31448.28 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-03-10 15:15:00)

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

### Cycle 10 — SELL (started 2026-03-12 10:15:00)

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

### Cycle 11 — BUY (started 2026-03-18 09:15:00)

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

### Cycle 12 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 31175.00 | 31492.48 | 31495.53 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-03-24 11:15:00)

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

### Cycle 14 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 31800.00 | 32200.04 | 32208.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 31525.00 | 31828.67 | 31994.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 32185.00 | 31880.95 | 31988.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 32535.00 | 31880.95 | 31988.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 32430.00 | 31990.76 | 32028.61 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2026-04-01 10:15:00)

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

### Cycle 16 — SELL (started 2026-04-23 09:15:00)

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

### Cycle 17 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 38130.00 | 37856.71 | 37829.86 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-04-24 12:15:00)

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

### Cycle 19 — BUY (started 2026-05-05 14:15:00)

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

### Cycle 20 — SELL (started 2026-05-08 10:15:00)

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
