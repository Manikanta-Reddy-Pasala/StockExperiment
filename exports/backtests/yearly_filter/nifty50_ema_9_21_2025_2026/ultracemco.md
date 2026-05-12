# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 59 |
| ALERT2 | 57 |
| ALERT2_SKIP | 26 |
| ALERT3 | 150 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 70 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 63
- **Target hits / Stop hits / Partials:** 0 / 72 / 1
- **Avg / median % per leg:** -0.31% / -0.64%
- **Sum % (uncompounded):** -22.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 6 | 14.6% | 0 | 41 | 0 | -0.30% | -12.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 41 | 6 | 14.6% | 0 | 41 | 0 | -0.30% | -12.3% |
| SELL (all) | 32 | 4 | 12.5% | 0 | 31 | 1 | -0.33% | -10.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.61% | -1.2% |
| SELL @ 3rd Alert (retest2) | 30 | 4 | 13.3% | 0 | 29 | 1 | -0.31% | -9.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.61% | -1.2% |
| retest2 (combined) | 71 | 10 | 14.1% | 0 | 70 | 1 | -0.31% | -21.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 11708.00 | 11580.11 | 11563.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 11722.00 | 11608.49 | 11577.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 11609.00 | 11634.63 | 11596.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 11665.00 | 11640.70 | 11602.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 11641.00 | 11640.70 | 11602.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 11682.00 | 11678.24 | 11648.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 11668.00 | 11678.24 | 11648.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 11650.00 | 11672.60 | 11648.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 11650.00 | 11672.60 | 11648.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 11674.00 | 11672.88 | 11651.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 11648.00 | 11672.88 | 11651.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 11698.00 | 11680.96 | 11658.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 11790.00 | 11687.37 | 11663.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 11831.00 | 11723.40 | 11684.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 11760.00 | 11861.69 | 11863.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 11760.00 | 11861.69 | 11863.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 11704.00 | 11830.16 | 11848.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 11824.00 | 11807.46 | 11833.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 11824.00 | 11807.46 | 11833.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 11884.00 | 11822.77 | 11838.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 11903.00 | 11822.77 | 11838.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 11781.00 | 11814.41 | 11833.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 11747.00 | 11814.41 | 11833.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 11750.00 | 11721.56 | 11731.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 11757.00 | 11739.76 | 11738.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 11757.00 | 11739.76 | 11738.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 11668.00 | 11730.35 | 11734.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 11458.00 | 11647.25 | 11689.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 11300.00 | 11299.74 | 11395.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:30:00 | 11321.00 | 11299.74 | 11395.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 11206.00 | 11146.09 | 11208.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 11206.00 | 11146.09 | 11208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 11165.00 | 11149.87 | 11204.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 11155.00 | 11149.87 | 11204.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 11119.00 | 11143.70 | 11196.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 11067.00 | 11123.69 | 11178.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 11065.00 | 11052.50 | 11080.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 11085.00 | 11071.00 | 11086.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 11154.00 | 11103.31 | 11098.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 11154.00 | 11103.31 | 11098.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 11205.00 | 11142.44 | 11119.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 11211.00 | 11219.11 | 11177.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 11211.00 | 11219.11 | 11177.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 11174.00 | 11209.75 | 11180.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 11172.00 | 11209.75 | 11180.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 11234.00 | 11214.60 | 11184.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 11266.00 | 11223.48 | 11191.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 11299.00 | 11381.73 | 11382.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 11299.00 | 11381.73 | 11382.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11333.98 | 11360.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:30:00 | 11331.00 | 11250.14 | 11289.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 11354.00 | 11270.91 | 11295.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 11355.00 | 11270.91 | 11295.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 11438.00 | 11330.90 | 11320.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 11493.00 | 11363.32 | 11335.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 11387.00 | 11406.14 | 11373.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 11384.00 | 11401.71 | 11374.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 11378.00 | 11401.71 | 11374.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 11397.00 | 11400.77 | 11376.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 11432.00 | 11394.93 | 11377.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 11411.00 | 11402.13 | 11385.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:00:00 | 11417.00 | 11405.11 | 11388.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:30:00 | 11408.00 | 11402.43 | 11390.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 11461.00 | 11413.75 | 11397.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 11482.00 | 11413.75 | 11397.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 11359.00 | 11415.81 | 11408.06 | SL hit (close<static) qty=1.00 sl=11376.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 11359.00 | 11412.26 | 11416.74 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 11474.00 | 11420.08 | 11418.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 11731.00 | 11488.29 | 11450.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 11572.00 | 11626.24 | 11548.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 11572.00 | 11626.24 | 11548.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 12364.00 | 12408.52 | 12359.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 12359.00 | 12408.52 | 12359.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 12360.00 | 12398.82 | 12359.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:30:00 | 12417.00 | 12399.05 | 12363.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 12329.00 | 12398.66 | 12381.21 | SL hit (close<static) qty=1.00 sl=12344.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 12470.00 | 12512.11 | 12515.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 12448.00 | 12501.30 | 12509.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 12524.00 | 12496.95 | 12504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 12494.00 | 12496.36 | 12503.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 12434.00 | 12497.83 | 12503.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 12483.00 | 12482.16 | 12484.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 12468.00 | 12479.33 | 12482.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 12520.00 | 12487.47 | 12486.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 12520.00 | 12487.47 | 12486.28 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 12470.00 | 12486.02 | 12487.22 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 12509.00 | 12491.25 | 12489.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 12636.00 | 12521.28 | 12503.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 15:15:00 | 12561.00 | 12567.17 | 12540.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 12565.00 | 12567.17 | 12540.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 12456.00 | 12544.93 | 12532.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 12456.00 | 12544.93 | 12532.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 12462.00 | 12528.35 | 12526.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 12369.00 | 12528.35 | 12526.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 12386.00 | 12499.88 | 12513.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 12289.00 | 12366.21 | 12415.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 12323.00 | 12306.39 | 12346.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 12:30:00 | 12317.00 | 12306.39 | 12346.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 12317.00 | 12285.08 | 12321.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 12293.00 | 12285.08 | 12321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 12262.00 | 12280.46 | 12316.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 12250.00 | 12268.66 | 12304.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 12355.00 | 12271.45 | 12268.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 12355.00 | 12271.45 | 12268.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 12361.00 | 12289.36 | 12276.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:30:00 | 12247.00 | 12286.11 | 12277.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 12251.00 | 12279.09 | 12274.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 12250.00 | 12279.09 | 12274.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 12201.00 | 12263.47 | 12268.22 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 12328.00 | 12277.49 | 12272.50 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 12250.00 | 12266.95 | 12268.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 12215.00 | 12256.56 | 12263.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 12138.00 | 12167.06 | 12200.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:00:00 | 12135.00 | 12167.06 | 12200.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 12289.00 | 12196.56 | 12208.57 | SL hit (close>static) qty=1.00 sl=12249.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 12327.00 | 12237.40 | 12225.50 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 12215.00 | 12260.20 | 12261.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 12196.00 | 12247.36 | 12255.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 12269.00 | 12218.45 | 12234.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 12292.00 | 12233.16 | 12239.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 12285.00 | 12233.16 | 12239.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 12183.00 | 12224.46 | 12234.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 12159.00 | 12208.29 | 12224.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 12150.00 | 12196.63 | 12217.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 12246.00 | 12201.77 | 12216.32 | SL hit (close>static) qty=1.00 sl=12242.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 12322.00 | 12241.53 | 12232.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 12389.00 | 12284.38 | 12254.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 12446.00 | 12447.88 | 12393.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 12446.00 | 12447.88 | 12393.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 12393.00 | 12439.43 | 12403.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 12393.00 | 12439.43 | 12403.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 12411.00 | 12433.75 | 12404.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 12370.00 | 12433.75 | 12404.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 12317.00 | 12410.40 | 12396.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 12317.00 | 12410.40 | 12396.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 12282.00 | 12384.72 | 12386.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 12251.00 | 12357.97 | 12373.83 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 12874.00 | 12431.80 | 12396.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 12923.00 | 12818.87 | 12708.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 12829.00 | 12857.70 | 12781.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 12791.00 | 12857.70 | 12781.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 12770.00 | 12845.42 | 12813.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 12755.00 | 12845.42 | 12813.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 12714.00 | 12819.14 | 12804.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 12714.00 | 12819.14 | 12804.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 12687.00 | 12792.71 | 12793.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 12640.00 | 12762.17 | 12779.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 12663.00 | 12638.25 | 12688.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 12663.00 | 12638.25 | 12688.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 12663.00 | 12610.78 | 12652.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 12663.00 | 12610.78 | 12652.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 12698.00 | 12628.23 | 12656.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 12698.00 | 12628.23 | 12656.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 12604.00 | 12614.81 | 12639.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 12641.00 | 12614.81 | 12639.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 12644.00 | 12595.72 | 12618.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 12660.00 | 12595.72 | 12618.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 12678.00 | 12612.17 | 12624.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 12678.00 | 12612.17 | 12624.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 12:15:00 | 12654.00 | 12633.63 | 12632.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 12706.00 | 12650.82 | 12641.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 12728.00 | 12756.65 | 12721.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 12733.00 | 12751.92 | 12722.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:00:00 | 12759.00 | 12745.07 | 12726.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 12750.00 | 12744.26 | 12727.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 12756.00 | 12744.26 | 12727.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 12684.00 | 12727.52 | 12722.62 | SL hit (close<static) qty=1.00 sl=12687.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 12650.00 | 12713.01 | 12719.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 12602.00 | 12680.09 | 12702.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 12620.00 | 12618.29 | 12661.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 12620.00 | 12618.29 | 12661.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 12695.00 | 12627.09 | 12654.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 12687.00 | 12627.09 | 12654.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 12634.00 | 12628.47 | 12652.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 12594.00 | 12642.03 | 12653.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 12515.00 | 12450.64 | 12442.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 12515.00 | 12450.64 | 12442.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 12558.00 | 12472.11 | 12453.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 12642.00 | 12672.74 | 12620.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 12642.00 | 12672.74 | 12620.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 12616.00 | 12661.39 | 12620.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 12593.00 | 12661.39 | 12620.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 12633.00 | 12655.71 | 12621.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 12572.00 | 12655.71 | 12621.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 12648.00 | 12654.17 | 12623.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 12542.00 | 12654.17 | 12623.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 12537.00 | 12630.74 | 12615.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 12537.00 | 12630.74 | 12615.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 12570.00 | 12618.59 | 12611.79 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 12567.00 | 12600.66 | 12604.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 12496.00 | 12573.54 | 12591.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 12664.00 | 12602.19 | 12598.31 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 12423.00 | 12594.79 | 12601.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 12354.00 | 12447.34 | 12508.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 12117.00 | 12102.03 | 12178.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:30:00 | 12052.00 | 12079.90 | 12155.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 12170.00 | 12077.37 | 12122.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12170.00 | 12077.37 | 12122.02 | SL hit (close>ema400) qty=1.00 sl=12122.02 alert=retest1 |

### Cycle 31 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 12205.00 | 12141.13 | 12140.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 12240.00 | 12174.16 | 12156.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 12135.00 | 12166.33 | 12154.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 12118.00 | 12156.67 | 12151.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 12118.00 | 12156.67 | 12151.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 12135.00 | 12145.99 | 12147.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 14:15:00 | 12090.00 | 12130.63 | 12139.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 12070.00 | 12040.74 | 12071.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 12085.00 | 12049.59 | 12072.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 12086.00 | 12049.59 | 12072.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 12092.00 | 12058.07 | 12074.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 12092.00 | 12058.07 | 12074.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 12077.00 | 12061.86 | 12074.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 12094.00 | 12061.86 | 12074.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 12050.00 | 12059.49 | 12072.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:15:00 | 12073.00 | 12059.49 | 12072.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 12073.00 | 12062.19 | 12072.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 12162.00 | 12062.19 | 12072.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 12178.00 | 12085.35 | 12081.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 12185.00 | 12121.07 | 12100.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 12113.00 | 12136.81 | 12114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 12039.00 | 12117.25 | 12107.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 12039.00 | 12117.25 | 12107.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 11968.00 | 12087.40 | 12095.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 11958.00 | 12061.52 | 12082.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 12055.00 | 12030.21 | 12057.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 12044.00 | 12032.97 | 12056.70 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 12177.00 | 12087.14 | 12075.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 12305.00 | 12142.85 | 12103.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 12089.00 | 12163.32 | 12168.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 12036.00 | 12137.86 | 12156.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 12191.00 | 12114.81 | 12133.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 12186.00 | 12129.05 | 12138.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 12186.00 | 12129.05 | 12138.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 12267.00 | 12168.63 | 12155.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 12288.00 | 12192.51 | 12167.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 12358.00 | 12336.55 | 12284.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 12362.00 | 12338.41 | 12297.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 12373.00 | 12333.87 | 12316.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 12388.00 | 12330.49 | 12316.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 12295.00 | 12323.39 | 12314.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 12295.00 | 12323.39 | 12314.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 12316.00 | 12321.91 | 12314.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 12293.00 | 12321.91 | 12314.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 12305.00 | 12318.53 | 12313.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 12300.00 | 12318.53 | 12313.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 12268.00 | 12308.43 | 12309.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 12268.00 | 12308.43 | 12309.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 12190.00 | 12284.74 | 12298.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 12:15:00 | 11991.00 | 11986.76 | 12068.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:00:00 | 11991.00 | 11986.76 | 12068.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 11971.00 | 11989.09 | 12044.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 11950.00 | 11981.42 | 12031.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 11957.00 | 11977.93 | 12025.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 12051.00 | 11985.75 | 12000.80 | SL hit (close>static) qty=1.00 sl=12048.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 12038.00 | 12006.89 | 12005.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 12100.00 | 12041.69 | 12024.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 12030.00 | 12039.35 | 12025.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 11991.00 | 12033.07 | 12026.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 11998.00 | 12033.07 | 12026.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 11947.00 | 12015.86 | 12018.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 11924.00 | 11997.48 | 12010.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 11957.00 | 11955.01 | 11981.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 11957.00 | 11955.01 | 11981.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 11952.00 | 11892.34 | 11918.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 11972.00 | 11892.34 | 11918.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 11946.00 | 11903.07 | 11921.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 11968.00 | 11903.07 | 11921.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 11911.00 | 11908.17 | 11920.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 11930.00 | 11908.17 | 11920.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 11942.00 | 11914.93 | 11922.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 11871.00 | 11914.93 | 11922.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:00:00 | 11895.00 | 11909.01 | 11916.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 11890.00 | 11905.21 | 11913.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 11890.00 | 11839.75 | 11834.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 11890.00 | 11839.75 | 11834.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 11899.00 | 11851.60 | 11840.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 11942.00 | 11943.44 | 11901.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 11942.00 | 11943.44 | 11901.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 11863.00 | 11925.05 | 11903.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 11863.00 | 11925.05 | 11903.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 11867.00 | 11913.44 | 11900.02 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 11819.00 | 11882.80 | 11887.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 11800.00 | 11859.58 | 11874.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 11763.00 | 11761.34 | 11797.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:45:00 | 11753.00 | 11761.34 | 11797.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 11698.00 | 11695.10 | 11731.34 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 11770.00 | 11741.41 | 11738.44 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 11717.00 | 11734.06 | 11735.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 11641.00 | 11715.45 | 11726.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 11635.00 | 11631.30 | 11666.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 11635.00 | 11631.30 | 11666.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 11628.00 | 11609.90 | 11641.19 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 11766.00 | 11675.54 | 11663.90 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 11614.00 | 11663.54 | 11666.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 11604.00 | 11639.48 | 11653.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 11592.00 | 11598.52 | 11620.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 11584.00 | 11592.05 | 11613.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 11585.00 | 11592.05 | 11613.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 11661.00 | 11612.00 | 11617.95 | SL hit (close>static) qty=1.00 sl=11651.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 12013.00 | 11692.20 | 11653.86 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 11559.00 | 11643.75 | 11644.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 11507.00 | 11574.77 | 11588.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 11472.00 | 11469.21 | 11510.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:30:00 | 11424.00 | 11450.69 | 11494.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 11419.00 | 11391.25 | 11429.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 11435.00 | 11391.25 | 11429.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 11451.00 | 11403.20 | 11431.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 11451.00 | 11403.20 | 11431.02 | SL hit (close>ema400) qty=1.00 sl=11431.02 alert=retest1 |

### Cycle 49 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 11629.00 | 11465.20 | 11453.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 11673.00 | 11557.10 | 11503.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 11661.00 | 11670.91 | 11595.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:30:00 | 11683.00 | 11670.91 | 11595.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 11619.00 | 11681.01 | 11631.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 11619.00 | 11681.01 | 11631.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 11617.00 | 11668.21 | 11630.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 11617.00 | 11668.21 | 11630.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 11571.00 | 11648.77 | 11624.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 11572.00 | 11648.77 | 11624.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 11520.00 | 11608.84 | 11611.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 11478.00 | 11529.79 | 11559.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 11496.00 | 11488.63 | 11520.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 11504.00 | 11493.44 | 11510.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 11475.00 | 11494.15 | 11509.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 11531.00 | 11503.93 | 11506.02 | SL hit (close>static) qty=1.00 sl=11529.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 11531.00 | 11509.35 | 11508.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 11675.00 | 11542.48 | 11523.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 11767.00 | 11792.41 | 11740.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 11766.00 | 11792.41 | 11740.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 11769.00 | 11787.73 | 11743.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 11804.00 | 11787.73 | 11743.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 11802.00 | 11790.58 | 11748.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 11800.00 | 11792.67 | 11753.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 11801.00 | 11791.33 | 11756.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 11754.00 | 11788.43 | 11766.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 11750.00 | 11788.43 | 11766.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 11702.00 | 11771.14 | 11760.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 11702.00 | 11771.14 | 11760.38 | SL hit (close<static) qty=1.00 sl=11731.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 11653.00 | 11747.51 | 11750.61 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 11766.00 | 11741.91 | 11740.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 11843.00 | 11780.22 | 11760.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 11834.00 | 11855.32 | 11820.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:00:00 | 11834.00 | 11855.32 | 11820.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 11888.00 | 11870.88 | 11837.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 11835.00 | 11870.88 | 11837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 12039.00 | 11914.68 | 11866.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 12092.00 | 11973.32 | 11902.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:00:00 | 12098.00 | 11973.32 | 11902.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 12096.00 | 11997.85 | 11920.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 12121.00 | 12030.86 | 11956.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 12122.00 | 12151.16 | 12100.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 12096.00 | 12151.16 | 12100.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 12069.00 | 12143.80 | 12110.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 12069.00 | 12143.80 | 12110.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 12108.00 | 12136.64 | 12110.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 12047.00 | 12088.79 | 12094.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 12047.00 | 12088.79 | 12094.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 11996.00 | 12070.24 | 12085.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 11930.00 | 11910.58 | 11969.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 11959.00 | 11910.58 | 11969.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 12055.00 | 11939.46 | 11977.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 12056.00 | 11939.46 | 11977.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 12080.00 | 11967.57 | 11986.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 12102.00 | 11967.57 | 11986.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 12064.00 | 12011.24 | 12004.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 12148.00 | 12062.32 | 12039.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 12320.00 | 12324.61 | 12242.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:30:00 | 12313.00 | 12324.61 | 12242.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 12261.00 | 12315.11 | 12276.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 12366.00 | 12315.11 | 12276.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 12360.00 | 12324.09 | 12283.66 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 12045.00 | 12225.09 | 12249.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 12026.00 | 12185.27 | 12229.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 12210.00 | 12169.02 | 12208.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 12239.00 | 12183.02 | 12211.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 12280.00 | 12183.02 | 12211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 12152.00 | 12176.81 | 12206.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 12133.00 | 12176.81 | 12206.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 12243.00 | 12190.05 | 12209.58 | SL hit (close>static) qty=1.00 sl=12239.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 12375.00 | 12225.43 | 12222.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 12450.00 | 12343.42 | 12292.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 12306.00 | 12392.70 | 12337.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 12371.00 | 12388.36 | 12340.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 12371.00 | 12388.36 | 12340.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 12346.00 | 12379.89 | 12340.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 12624.00 | 12379.89 | 12340.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 12514.00 | 12637.43 | 12642.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 12514.00 | 12637.43 | 12642.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 12285.00 | 12551.99 | 12601.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 12429.00 | 12406.30 | 12497.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 12429.00 | 12406.30 | 12497.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 12449.00 | 12414.84 | 12493.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:30:00 | 12478.00 | 12414.84 | 12493.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 12531.00 | 12438.08 | 12496.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 12531.00 | 12438.08 | 12496.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 12543.00 | 12459.06 | 12500.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 12680.00 | 12459.06 | 12500.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 12745.00 | 12550.16 | 12537.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 12775.00 | 12635.62 | 12592.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 12767.00 | 12770.58 | 12717.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:45:00 | 12771.00 | 12770.58 | 12717.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 12706.00 | 12757.57 | 12721.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 12665.00 | 12757.57 | 12721.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 12718.00 | 12749.66 | 12720.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 12705.00 | 12749.66 | 12720.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 12695.00 | 12738.73 | 12718.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 12695.00 | 12738.73 | 12718.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 12643.00 | 12719.58 | 12711.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 12640.00 | 12719.58 | 12711.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 12733.00 | 12719.06 | 12712.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 12766.00 | 12719.06 | 12712.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 12821.00 | 12737.05 | 12721.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 12912.00 | 12950.32 | 12952.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 12912.00 | 12950.32 | 12952.39 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 12984.00 | 12947.07 | 12943.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 13025.00 | 12968.57 | 12954.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 13000.00 | 13016.74 | 12990.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 12955.00 | 13004.39 | 12987.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 12955.00 | 13004.39 | 12987.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 12834.00 | 12970.31 | 12973.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 12707.00 | 12896.32 | 12937.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 12802.00 | 12784.55 | 12843.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 12802.00 | 12784.55 | 12843.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 12822.00 | 12783.82 | 12827.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 12826.00 | 12783.82 | 12827.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 12811.00 | 12789.25 | 12826.31 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 12964.00 | 12865.93 | 12854.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 13056.00 | 12975.63 | 12941.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 12980.00 | 12988.32 | 12957.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 12963.00 | 12983.26 | 12957.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 12963.00 | 12983.26 | 12957.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 12923.00 | 12971.21 | 12954.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 12932.00 | 12971.21 | 12954.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 12873.00 | 12951.56 | 12947.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 12873.00 | 12951.56 | 12947.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 12933.00 | 12942.96 | 12943.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 12669.00 | 12888.17 | 12918.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 12541.00 | 12539.06 | 12663.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 12541.00 | 12539.06 | 12663.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 12303.00 | 12149.81 | 12257.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 12303.00 | 12149.81 | 12257.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 12260.00 | 12171.85 | 12257.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 12150.00 | 12171.85 | 12257.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 11542.50 | 11980.98 | 12112.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 11631.00 | 11550.52 | 11775.17 | SL hit (close>ema200) qty=0.50 sl=11550.52 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 11119.00 | 11058.54 | 11054.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 11206.00 | 11088.03 | 11068.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 10955.00 | 11149.98 | 11124.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 10906.00 | 11101.19 | 11104.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 10886.00 | 11029.32 | 11069.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 10929.00 | 10924.27 | 10976.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 10922.00 | 10923.81 | 10971.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 10922.00 | 10923.81 | 10971.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 10922.00 | 10923.81 | 10971.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 10549.00 | 10513.90 | 10681.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 10502.00 | 10516.32 | 10667.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 10823.00 | 10588.81 | 10674.79 | SL hit (close>static) qty=1.00 sl=10685.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 11094.00 | 10764.84 | 10737.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 11158.00 | 10843.47 | 10775.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 11030.00 | 11069.64 | 10941.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:30:00 | 11006.00 | 11069.64 | 10941.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 10937.00 | 11046.44 | 10994.52 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 10789.00 | 10942.03 | 10954.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 10750.00 | 10903.62 | 10936.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 10867.00 | 10840.06 | 10886.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 10867.00 | 10840.06 | 10886.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 10848.00 | 10841.65 | 10882.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 10800.00 | 10841.65 | 10882.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 10928.00 | 10731.87 | 10727.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 10928.00 | 10731.87 | 10727.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 10963.00 | 10778.09 | 10748.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 10796.00 | 10816.82 | 10777.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 11325.00 | 11503.10 | 11439.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 11427.00 | 11487.88 | 11438.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 11443.00 | 11482.70 | 11440.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 12035.00 | 12068.63 | 12069.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 12035.00 | 12068.63 | 12069.07 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 12096.00 | 12074.11 | 12071.51 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 15:15:00 | 12000.00 | 12079.01 | 12080.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 11909.00 | 12045.00 | 12065.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 11958.00 | 11898.85 | 11962.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:00:00 | 11958.00 | 11898.85 | 11962.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 11996.00 | 11918.28 | 11965.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:15:00 | 12016.00 | 11918.28 | 11965.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 12024.00 | 11939.42 | 11970.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 12024.00 | 11939.42 | 11970.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 11853.00 | 11927.03 | 11959.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 11838.00 | 11906.42 | 11947.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 11921.00 | 11775.13 | 11761.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 11921.00 | 11775.13 | 11761.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 13:15:00 | 11946.00 | 11809.31 | 11778.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 12121.00 | 12127.69 | 12036.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 12092.00 | 12127.69 | 12036.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 11964.00 | 12094.95 | 12029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 11964.00 | 12094.95 | 12029.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 11955.00 | 12066.96 | 12022.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:15:00 | 11907.00 | 12066.96 | 12022.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 11956.00 | 11994.86 | 11997.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 11930.00 | 11981.89 | 11991.25 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:30:00 | 11790.00 | 2025-05-20 13:15:00 | 11760.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-05-15 13:00:00 | 11831.00 | 2025-05-20 13:15:00 | 11760.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-21 12:15:00 | 11747.00 | 2025-05-23 13:15:00 | 11757.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-05-23 11:45:00 | 11750.00 | 2025-05-23 13:15:00 | 11757.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-06-03 11:30:00 | 11067.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-05 11:00:00 | 11065.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-05 11:30:00 | 11085.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-09 15:15:00 | 11266.00 | 2025-06-12 15:15:00 | 11299.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-18 09:15:00 | 11432.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-18 12:00:00 | 11411.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-18 13:00:00 | 11417.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-18 14:30:00 | 11408.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-19 10:15:00 | 11482.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-20 09:15:00 | 11491.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-06-20 10:15:00 | 11476.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-20 10:45:00 | 11478.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-20 15:15:00 | 11475.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-04 13:30:00 | 12417.00 | 2025-07-07 11:15:00 | 12329.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-07 12:30:00 | 12389.00 | 2025-07-07 14:15:00 | 12343.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-07 13:45:00 | 12370.00 | 2025-07-07 14:15:00 | 12343.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-07-08 09:15:00 | 12376.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-07-09 13:45:00 | 12524.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-11 09:30:00 | 12625.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-07-11 14:30:00 | 12496.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-07-14 09:30:00 | 12570.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-16 09:15:00 | 12434.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-17 10:30:00 | 12483.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-17 12:00:00 | 12468.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-28 12:30:00 | 12250.00 | 2025-07-30 11:15:00 | 12355.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-04 11:30:00 | 12138.00 | 2025-08-04 13:15:00 | 12289.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-04 12:00:00 | 12135.00 | 2025-08-04 13:15:00 | 12289.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-08 14:15:00 | 12159.00 | 2025-08-11 09:15:00 | 12246.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-08 15:00:00 | 12150.00 | 2025-08-11 09:15:00 | 12246.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-09-03 11:00:00 | 12759.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-03 11:45:00 | 12750.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-03 12:15:00 | 12756.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-04 09:15:00 | 12930.00 | 2025-09-04 14:15:00 | 12650.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-09-09 13:00:00 | 12594.00 | 2025-09-16 10:15:00 | 12515.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest1 | 2025-09-29 11:30:00 | 12052.00 | 2025-09-30 09:15:00 | 12170.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-17 11:15:00 | 12358.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-17 14:00:00 | 12362.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-21 13:45:00 | 12373.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-23 09:15:00 | 12388.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-28 12:00:00 | 11950.00 | 2025-10-29 12:15:00 | 12051.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-28 13:15:00 | 11957.00 | 2025-10-29 12:15:00 | 12051.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-30 09:30:00 | 11955.00 | 2025-10-30 11:15:00 | 12038.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-11-07 09:15:00 | 11871.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-07 13:00:00 | 11895.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-11-07 14:00:00 | 11890.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-01 10:15:00 | 11592.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-01 11:30:00 | 11584.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-01 12:00:00 | 11585.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-12-10 11:30:00 | 11424.00 | 2025-12-11 13:15:00 | 11451.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-22 09:15:00 | 11475.00 | 2025-12-22 14:15:00 | 11531.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-29 10:15:00 | 11804.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-29 11:00:00 | 11802.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-29 11:45:00 | 11800.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-29 12:30:00 | 11801.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-05 11:30:00 | 12092.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2026-01-05 12:00:00 | 12098.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-01-05 13:00:00 | 12096.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-01-06 09:15:00 | 12121.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-21 14:15:00 | 12133.00 | 2026-01-21 14:15:00 | 12243.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-27 09:15:00 | 12624.00 | 2026-02-01 12:15:00 | 12514.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-09 09:15:00 | 12766.00 | 2026-02-16 10:15:00 | 12912.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2026-02-09 09:45:00 | 12821.00 | 2026-02-16 10:15:00 | 12912.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2026-03-06 09:15:00 | 12150.00 | 2026-03-09 09:15:00 | 11542.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 12150.00 | 2026-03-10 09:15:00 | 11631.00 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2026-03-24 10:30:00 | 10502.00 | 2026-03-24 12:15:00 | 10823.00 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-04-01 13:15:00 | 10800.00 | 2026-04-06 13:15:00 | 10928.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-13 11:00:00 | 11427.00 | 2026-04-24 15:15:00 | 12035.00 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2026-04-13 11:30:00 | 11443.00 | 2026-04-24 15:15:00 | 12035.00 | STOP_HIT | 1.00 | 5.17% |
| SELL | retest2 | 2026-04-29 14:45:00 | 11838.00 | 2026-05-05 12:15:00 | 11921.00 | STOP_HIT | 1.00 | -0.70% |
