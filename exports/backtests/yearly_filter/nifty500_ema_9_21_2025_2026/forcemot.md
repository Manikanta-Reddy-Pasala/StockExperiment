# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 20851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 51 |
| ALERT2 | 51 |
| ALERT2_SKIP | 24 |
| ALERT3 | 112 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 34 |
| PARTIAL | 6 |
| TARGET_HIT | 8 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 22
- **Target hits / Stop hits / Partials:** 8 / 30 / 6
- **Avg / median % per leg:** 1.88% / 0.02%
- **Sum % (uncompounded):** 82.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 12 | 42.9% | 8 | 19 | 1 | 2.06% | 57.7% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.73% | 14.2% |
| BUY @ 3rd Alert (retest2) | 25 | 10 | 40.0% | 7 | 18 | 0 | 1.74% | 43.5% |
| SELL (all) | 16 | 10 | 62.5% | 0 | 11 | 5 | 1.55% | 24.9% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.82% | 11.3% |
| SELL @ 3rd Alert (retest2) | 12 | 6 | 50.0% | 0 | 9 | 3 | 1.13% | 13.6% |
| retest1 (combined) | 7 | 6 | 85.7% | 1 | 3 | 3 | 3.64% | 25.5% |
| retest2 (combined) | 37 | 16 | 43.2% | 7 | 27 | 3 | 1.54% | 57.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 10730.00 | 10036.46 | 9962.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 10754.00 | 10411.38 | 10184.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 10310.00 | 10469.88 | 10254.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 10544.50 | 10564.34 | 10424.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 10544.50 | 10564.34 | 10424.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 10492.00 | 10549.87 | 10430.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 10490.00 | 10549.87 | 10430.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 10411.00 | 10516.34 | 10444.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 10411.00 | 10516.34 | 10444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 10420.50 | 10497.17 | 10442.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 10549.00 | 10497.17 | 10442.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 10605.50 | 10767.08 | 10772.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 10605.50 | 10767.08 | 10772.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 10595.00 | 10684.93 | 10721.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 10510.00 | 10507.82 | 10592.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:00:00 | 10510.00 | 10507.82 | 10592.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 10599.00 | 10526.06 | 10592.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:30:00 | 10620.00 | 10526.06 | 10592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 10561.00 | 10533.05 | 10590.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 10550.00 | 10533.05 | 10590.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 10527.50 | 10531.94 | 10584.32 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 10840.00 | 10617.88 | 10597.16 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 10203.50 | 10526.85 | 10564.66 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 10933.50 | 10611.96 | 10569.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 11030.00 | 10827.40 | 10701.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 12163.00 | 12446.37 | 12142.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:00:00 | 12163.00 | 12446.37 | 12142.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 11960.00 | 12349.10 | 12125.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 11960.00 | 12349.10 | 12125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 11870.00 | 12253.28 | 12102.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 11870.00 | 12253.28 | 12102.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 11652.00 | 12011.15 | 12018.54 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 12100.00 | 11990.02 | 11989.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 12278.00 | 12061.24 | 12023.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 15:15:00 | 12244.00 | 12256.36 | 12151.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:15:00 | 12498.00 | 12256.36 | 12151.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 12397.00 | 12579.39 | 12420.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 12397.00 | 12579.39 | 12420.05 | SL hit (close<ema400) qty=1.00 sl=12420.05 alert=retest1 |

### Cycle 8 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 12234.00 | 12405.57 | 12421.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 12169.00 | 12358.26 | 12398.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 12562.00 | 12321.10 | 12360.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 12390.00 | 12334.88 | 12362.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:30:00 | 12328.00 | 12306.90 | 12347.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 12328.00 | 12213.87 | 12238.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 12687.00 | 12308.50 | 12279.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 12687.00 | 12308.50 | 12279.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 13120.00 | 12470.80 | 12355.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 14000.00 | 14063.88 | 13698.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 14000.00 | 14063.88 | 13698.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 13696.00 | 13981.52 | 13723.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 14180.00 | 14062.30 | 13806.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 13659.00 | 13789.65 | 13801.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 13659.00 | 13789.65 | 13801.66 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 14000.00 | 13843.93 | 13825.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 14185.00 | 13912.15 | 13857.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 14125.00 | 14228.10 | 14130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 14221.00 | 14226.68 | 14138.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 14149.00 | 14226.68 | 14138.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 14120.00 | 14205.34 | 14136.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 14120.00 | 14205.34 | 14136.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 14137.00 | 14191.68 | 14136.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 14137.00 | 14191.68 | 14136.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 13650.00 | 14083.34 | 14092.71 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 14260.00 | 13981.52 | 13968.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 14546.00 | 14218.45 | 14102.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 15335.00 | 15535.13 | 15145.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:00:00 | 15335.00 | 15535.13 | 15145.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 15071.00 | 15442.31 | 15139.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 15048.00 | 15442.31 | 15139.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 14767.00 | 15307.25 | 15105.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 14737.00 | 15307.25 | 15105.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 14779.00 | 15201.60 | 15075.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 14968.00 | 15201.60 | 15075.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 14711.00 | 14991.95 | 15001.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 14711.00 | 14991.95 | 15001.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 14515.00 | 14686.71 | 14810.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 14390.00 | 14356.08 | 14478.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 14390.00 | 14356.08 | 14478.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 14796.00 | 14444.06 | 14507.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 14796.00 | 14444.06 | 14507.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 14859.00 | 14527.05 | 14539.36 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 14792.00 | 14580.04 | 14562.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 10:15:00 | 14902.00 | 14735.08 | 14650.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 16507.00 | 16544.91 | 16183.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:15:00 | 16392.00 | 16544.91 | 16183.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 16252.00 | 16428.24 | 16215.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 16398.00 | 16396.55 | 16236.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 16615.00 | 16804.74 | 16829.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 16615.00 | 16804.74 | 16829.32 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 17047.00 | 16864.96 | 16841.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 17090.00 | 16909.97 | 16864.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 17329.00 | 17398.95 | 17253.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:45:00 | 17327.00 | 17398.95 | 17253.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 17110.00 | 17341.16 | 17240.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 17133.00 | 17341.16 | 17240.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 17195.00 | 17311.93 | 17236.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:15:00 | 17103.00 | 17311.93 | 17236.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 17112.00 | 17271.94 | 17225.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 19005.00 | 17259.55 | 17223.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:45:00 | 17380.00 | 18417.36 | 18170.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 17823.00 | 18034.83 | 18062.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 17823.00 | 18034.83 | 18062.89 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 18270.00 | 18081.86 | 18081.72 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 17951.00 | 18103.28 | 18118.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 14:15:00 | 17807.00 | 18044.03 | 18089.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 17047.00 | 16884.13 | 17143.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 17045.00 | 16999.85 | 17087.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 17008.00 | 17032.28 | 17093.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 17422.00 | 17110.22 | 17123.78 | SL hit (close>static) qty=1.00 sl=17280.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 17463.00 | 17180.78 | 17154.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 17642.00 | 17273.02 | 17198.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 17804.00 | 17957.66 | 17716.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 17804.00 | 17957.66 | 17716.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 17810.00 | 17985.50 | 17906.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 17810.00 | 17985.50 | 17906.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 17750.00 | 17938.40 | 17892.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 18091.00 | 17938.40 | 17892.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-13 13:15:00 | 19900.10 | 19379.17 | 18929.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 20443.00 | 20829.81 | 20844.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 20188.00 | 20701.45 | 20784.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 19948.00 | 19666.14 | 19892.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 20176.00 | 19768.11 | 19918.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 20176.00 | 19768.11 | 19918.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 19961.00 | 19806.69 | 19922.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 19829.00 | 19813.75 | 19914.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:15:00 | 18837.55 | 19414.05 | 19651.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 19285.00 | 19253.24 | 19467.98 | SL hit (close>ema200) qty=0.50 sl=19253.24 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 19930.00 | 19491.25 | 19490.82 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 19042.00 | 19480.59 | 19532.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 18100.00 | 18910.51 | 19221.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 17616.00 | 17428.24 | 17891.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:15:00 | 17668.00 | 17428.24 | 17891.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 17908.00 | 17618.43 | 17870.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 17807.00 | 17618.43 | 17870.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 17868.00 | 17668.34 | 17870.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 17900.00 | 17668.34 | 17870.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 18150.00 | 17764.67 | 17896.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 18150.00 | 17764.67 | 17896.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 18325.00 | 17876.74 | 17935.00 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 18510.00 | 18052.55 | 18007.85 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 18000.00 | 18104.18 | 18113.21 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 18347.00 | 18150.80 | 18132.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 18417.00 | 18204.04 | 18158.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 19168.00 | 19609.23 | 19388.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 19453.00 | 19577.99 | 19394.06 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 19000.00 | 19310.37 | 19314.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 15:15:00 | 18930.00 | 19234.29 | 19279.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 19260.00 | 19234.64 | 19275.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 19143.00 | 19216.31 | 19263.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 19036.00 | 19154.05 | 19230.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 15:15:00 | 18084.20 | 18938.23 | 19090.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 18205.00 | 18198.27 | 18475.01 | SL hit (close>ema200) qty=0.50 sl=18198.27 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 12:15:00 | 15892.00 | 15708.72 | 15686.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 16060.00 | 15778.98 | 15720.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 16590.00 | 16602.35 | 16369.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:30:00 | 16582.00 | 16602.35 | 16369.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 16530.00 | 16636.65 | 16532.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 17162.00 | 16636.65 | 16532.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 16850.00 | 17180.64 | 17224.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 16850.00 | 17180.64 | 17224.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 16400.00 | 17024.51 | 17149.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 16379.00 | 16374.67 | 16567.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 16379.00 | 16374.67 | 16567.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 16380.00 | 16370.46 | 16479.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 16658.00 | 16370.46 | 16479.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 16750.00 | 16446.37 | 16503.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 16750.00 | 16446.37 | 16503.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 16566.00 | 16470.29 | 16509.58 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 16650.00 | 16534.03 | 16533.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 16757.00 | 16624.54 | 16580.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 18130.00 | 18235.77 | 17893.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:00:00 | 18130.00 | 18235.77 | 17893.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 18157.00 | 18304.78 | 18138.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 18122.00 | 18304.78 | 18138.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 17879.00 | 18219.63 | 18115.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 17879.00 | 18219.63 | 18115.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 17890.00 | 18153.70 | 18094.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 17725.00 | 18153.70 | 18094.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 17930.00 | 18046.61 | 18052.40 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 18265.00 | 18090.29 | 18071.73 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 17610.00 | 18053.57 | 18093.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 17510.00 | 17944.86 | 18040.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 17864.00 | 17702.17 | 17850.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 17838.00 | 17729.33 | 17849.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 17905.00 | 17729.33 | 17849.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 17701.00 | 17723.67 | 17835.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:30:00 | 17664.00 | 17689.93 | 17810.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 17725.00 | 17553.44 | 17533.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 17725.00 | 17553.44 | 17533.65 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 17300.00 | 17521.24 | 17533.13 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 17621.00 | 17522.49 | 17513.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 17654.00 | 17548.79 | 17525.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 17461.00 | 17536.83 | 17524.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 17509.00 | 17531.26 | 17523.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 17585.00 | 17552.01 | 17533.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 17612.00 | 17548.72 | 17537.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 15:15:00 | 17464.00 | 17531.78 | 17530.45 | SL hit (close<static) qty=1.00 sl=17470.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 17289.00 | 17483.22 | 17508.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 17196.00 | 17425.78 | 17480.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 16808.00 | 16764.53 | 16920.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 16808.00 | 16764.53 | 16920.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 17006.00 | 16821.53 | 16908.10 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 17056.00 | 16946.73 | 16941.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 17182.00 | 16993.78 | 16963.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 18035.00 | 18121.95 | 17894.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 18035.00 | 18121.95 | 17894.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 17959.00 | 18089.36 | 17900.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 17923.00 | 18089.36 | 17900.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 17948.00 | 18061.09 | 17904.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 17948.00 | 18061.09 | 17904.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 17906.00 | 18030.07 | 17904.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:15:00 | 17850.00 | 18030.07 | 17904.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 17850.00 | 17994.05 | 17899.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 17602.00 | 17994.05 | 17899.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 17570.00 | 17909.24 | 17869.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 17570.00 | 17909.24 | 17869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 17535.00 | 17834.39 | 17839.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 17168.00 | 17568.34 | 17664.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 17328.00 | 17286.49 | 17438.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 17328.00 | 17286.49 | 17438.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 17190.00 | 17190.42 | 17295.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 17266.00 | 17190.42 | 17295.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 17419.00 | 17236.13 | 17307.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 17419.00 | 17236.13 | 17307.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 17229.00 | 17234.71 | 17300.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 17134.00 | 17220.41 | 17282.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 17352.00 | 17055.27 | 17030.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 17352.00 | 17055.27 | 17030.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 17515.00 | 17147.22 | 17074.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 17300.00 | 17321.33 | 17215.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 17300.00 | 17321.33 | 17215.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 17200.00 | 17297.06 | 17214.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 17200.00 | 17297.06 | 17214.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 17199.00 | 17277.45 | 17213.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 17190.00 | 17277.45 | 17213.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 17342.00 | 17283.89 | 17226.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 17393.00 | 17287.31 | 17233.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 17406.00 | 17319.96 | 17262.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 17348.00 | 17315.22 | 17284.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 17181.00 | 17288.38 | 17275.24 | SL hit (close<static) qty=1.00 sl=17219.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 17205.00 | 17259.00 | 17263.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 17100.00 | 17227.20 | 17248.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 17220.00 | 17125.58 | 17178.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 17309.00 | 17162.26 | 17189.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 17309.00 | 17162.26 | 17189.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 17302.00 | 17219.90 | 17211.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 18126.00 | 17401.12 | 17295.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 18376.00 | 18393.70 | 18256.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 18376.00 | 18393.70 | 18256.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 18886.00 | 18551.16 | 18426.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 19189.00 | 18690.73 | 18501.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 19038.00 | 18850.02 | 18629.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 19081.00 | 18877.73 | 18680.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 19010.00 | 18904.19 | 18710.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-31 09:15:00 | 20941.80 | 19568.47 | 19158.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 20770.00 | 20817.87 | 20823.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 20580.00 | 20735.33 | 20778.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 19750.00 | 19263.25 | 19621.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 19790.00 | 19368.60 | 19636.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 19815.00 | 19368.60 | 19636.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 19975.00 | 19787.33 | 19769.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 20370.00 | 20008.81 | 19891.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 20325.00 | 20327.32 | 20145.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 20325.00 | 20327.32 | 20145.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 20415.00 | 20324.49 | 20174.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 20560.00 | 20377.18 | 20236.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 19980.00 | 20624.99 | 20685.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 19980.00 | 20624.99 | 20685.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 19590.00 | 20190.79 | 20452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 20475.00 | 20148.89 | 20358.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 20465.00 | 20212.11 | 20368.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 20550.00 | 20212.11 | 20368.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 20455.00 | 20260.69 | 20376.09 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 20655.00 | 20450.87 | 20444.25 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 19930.00 | 20351.70 | 20404.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 19395.00 | 20014.27 | 20224.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 19100.00 | 18650.14 | 18863.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 19290.00 | 18778.11 | 18902.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 19290.00 | 18778.11 | 18902.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 19350.00 | 19035.43 | 19002.04 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 18686.00 | 19017.44 | 19037.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 18601.00 | 18929.76 | 18993.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 18897.00 | 18809.52 | 18912.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 18897.00 | 18809.52 | 18912.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 19221.00 | 18882.93 | 18927.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 19221.00 | 18882.93 | 18927.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 19300.00 | 18966.34 | 18961.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 19754.00 | 19123.88 | 19033.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 15:15:00 | 21197.00 | 21260.99 | 20898.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:15:00 | 21881.00 | 21260.99 | 20898.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:15:00 | 22975.05 | 22323.81 | 21716.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-11 09:15:00 | 24069.10 | 23478.82 | 22689.07 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 52 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 24308.00 | 24993.18 | 25003.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 24168.00 | 24648.19 | 24828.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 24335.00 | 24265.30 | 24557.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:30:00 | 24275.00 | 24265.30 | 24557.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 24475.00 | 24290.67 | 24493.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 24482.00 | 24290.67 | 24493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 24385.00 | 24309.54 | 24483.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 24500.00 | 24309.54 | 24483.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 24467.00 | 24341.03 | 24482.07 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 25029.00 | 24520.56 | 24500.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 25145.00 | 24863.49 | 24705.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 25418.00 | 25425.45 | 25229.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 25200.00 | 25380.36 | 25227.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 25200.00 | 25380.36 | 25227.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 25320.00 | 25380.36 | 25227.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 24782.00 | 25260.69 | 25186.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 24782.00 | 25260.69 | 25186.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 24554.00 | 25119.35 | 25129.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 24440.00 | 24983.48 | 25066.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 22415.00 | 22090.65 | 22902.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 14:30:00 | 21570.00 | 22173.64 | 22655.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 21305.00 | 22101.91 | 22579.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 20491.50 | 21181.12 | 21754.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 20239.75 | 21181.12 | 21754.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 21300.00 | 20879.76 | 21262.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 21300.00 | 20879.76 | 21262.01 | SL hit (close>ema200) qty=0.50 sl=20879.76 alert=retest1 |

### Cycle 55 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 22320.00 | 21500.29 | 21460.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 22685.00 | 21945.15 | 21691.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 22000.00 | 22060.39 | 21819.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:30:00 | 22135.00 | 22060.39 | 21819.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 21650.00 | 21948.65 | 21808.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 21650.00 | 21948.65 | 21808.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 21640.00 | 21886.92 | 21793.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 21050.00 | 21886.92 | 21793.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 21380.00 | 21667.63 | 21702.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 20880.00 | 21499.47 | 21615.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 20790.00 | 20479.55 | 20802.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 20750.00 | 20533.64 | 20797.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 20430.00 | 20533.64 | 20797.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 21085.00 | 20645.73 | 20803.14 | SL hit (close>static) qty=1.00 sl=20800.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 21040.00 | 20890.63 | 20880.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 21610.00 | 21034.50 | 20946.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 20885.00 | 21152.07 | 21162.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 20605.00 | 21042.66 | 21112.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 21565.00 | 21162.68 | 21148.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 21750.00 | 21395.71 | 21276.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 20625.00 | 21241.57 | 21216.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 20525.00 | 21098.25 | 21153.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 20415.00 | 20857.88 | 21028.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 21460.00 | 21000.69 | 20987.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 21960.00 | 21378.01 | 21186.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 21510.00 | 21649.01 | 21421.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 21510.00 | 21649.01 | 21421.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 21350.00 | 21589.21 | 21415.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 20680.00 | 21589.21 | 21415.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 20725.00 | 21416.37 | 21352.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 20600.00 | 21416.37 | 21352.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 20620.00 | 21257.09 | 21285.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 20545.00 | 21114.67 | 21218.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 20850.00 | 20435.90 | 20426.82 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 20305.00 | 20409.72 | 20415.74 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 20758.00 | 20469.71 | 20438.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 21085.00 | 20592.77 | 20497.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:30:00 | 20838.00 | 20713.18 | 20603.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 20315.00 | 20755.04 | 20694.15 | SL hit (close<static) qty=1.00 sl=20459.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 20301.00 | 20596.87 | 20628.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 20093.00 | 20496.09 | 20579.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 20950.00 | 20421.92 | 20509.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 21440.00 | 20625.54 | 20594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 21701.00 | 20840.63 | 20694.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 21945.00 | 22061.97 | 21708.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 21945.00 | 22061.97 | 21708.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 22150.00 | 22306.78 | 22007.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 22280.00 | 22306.78 | 22007.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 22983.00 | 22270.47 | 22130.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:15:00 | 22230.00 | 22357.67 | 22324.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 22041.00 | 22342.43 | 22365.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 22041.00 | 22342.43 | 22365.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 21862.00 | 21985.76 | 22103.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 21914.00 | 21910.79 | 22024.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:30:00 | 21939.00 | 21910.79 | 22024.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 21944.00 | 21917.43 | 22017.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 21792.00 | 21892.35 | 21997.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 14:15:00 | 20702.40 | 21087.03 | 21470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 20800.00 | 20529.79 | 20865.35 | SL hit (close>ema200) qty=0.50 sl=20529.79 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 21194.00 | 20614.34 | 20585.20 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 19626.00 | 20540.81 | 20630.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 19561.00 | 19963.41 | 20243.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 20342.00 | 19398.22 | 19567.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 20018.00 | 19522.17 | 19608.13 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 20057.00 | 19723.19 | 19690.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 20151.00 | 19808.75 | 19732.28 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 10549.00 | 2025-05-19 13:15:00 | 10605.50 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest1 | 2025-06-05 09:15:00 | 12498.00 | 2025-06-06 09:15:00 | 12397.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-10 11:30:00 | 12328.00 | 2025-06-12 11:15:00 | 12687.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-06-12 11:00:00 | 12328.00 | 2025-06-12 11:15:00 | 12687.00 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-06-18 11:30:00 | 14180.00 | 2025-06-19 15:15:00 | 13659.00 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-07-02 09:15:00 | 14968.00 | 2025-07-02 11:15:00 | 14711.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-11 14:30:00 | 16398.00 | 2025-07-18 14:15:00 | 16615.00 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-07-24 09:15:00 | 19005.00 | 2025-07-28 09:15:00 | 17823.00 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2025-07-25 10:45:00 | 17380.00 | 2025-07-28 09:15:00 | 17823.00 | STOP_HIT | 1.00 | 2.55% |
| SELL | retest2 | 2025-08-05 10:30:00 | 17008.00 | 2025-08-05 11:15:00 | 17422.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-08-11 09:15:00 | 18091.00 | 2025-08-13 13:15:00 | 19900.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 12:45:00 | 19829.00 | 2025-09-01 11:15:00 | 18837.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 12:45:00 | 19829.00 | 2025-09-02 09:15:00 | 19285.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-09-19 11:45:00 | 19036.00 | 2025-09-19 15:15:00 | 18084.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 11:45:00 | 19036.00 | 2025-09-23 13:15:00 | 18205.00 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2025-10-17 09:15:00 | 17162.00 | 2025-10-23 14:15:00 | 16850.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-12 12:30:00 | 17664.00 | 2025-11-17 09:15:00 | 17725.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-11-20 11:45:00 | 17585.00 | 2025-11-20 15:15:00 | 17464.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-11-20 14:30:00 | 17612.00 | 2025-11-20 15:15:00 | 17464.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-09 14:30:00 | 17134.00 | 2025-12-12 10:15:00 | 17352.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-16 09:15:00 | 17393.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-16 11:45:00 | 17406.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-17 09:30:00 | 17348.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-29 10:45:00 | 19189.00 | 2025-12-31 09:15:00 | 20941.80 | TARGET_HIT | 1.00 | 9.13% |
| BUY | retest2 | 2025-12-29 13:45:00 | 19038.00 | 2025-12-31 09:15:00 | 20911.00 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2025-12-30 09:15:00 | 19081.00 | 2026-01-02 09:15:00 | 21107.90 | TARGET_HIT | 1.00 | 10.62% |
| BUY | retest2 | 2025-12-30 10:00:00 | 19010.00 | 2026-01-02 09:15:00 | 20989.10 | TARGET_HIT | 1.00 | 10.41% |
| BUY | retest2 | 2026-01-07 09:45:00 | 21005.00 | 2026-01-07 10:15:00 | 20770.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-16 15:00:00 | 20560.00 | 2026-01-21 10:15:00 | 19980.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2026-02-09 09:15:00 | 21881.00 | 2026-02-10 09:15:00 | 22975.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-09 09:15:00 | 21881.00 | 2026-02-11 09:15:00 | 24069.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-13 12:15:00 | 23868.00 | 2026-02-18 09:15:00 | 26254.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 12:45:00 | 23741.00 | 2026-02-18 09:15:00 | 26115.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-03-05 14:30:00 | 21570.00 | 2026-03-09 09:15:00 | 20491.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-06 09:15:00 | 21305.00 | 2026-03-09 09:15:00 | 20239.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 14:30:00 | 21570.00 | 2026-03-10 09:15:00 | 21300.00 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2026-03-06 09:15:00 | 21305.00 | 2026-03-10 09:15:00 | 21300.00 | STOP_HIT | 0.50 | 0.02% |
| SELL | retest2 | 2026-03-17 09:15:00 | 20430.00 | 2026-03-17 10:15:00 | 21085.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-04-06 12:30:00 | 20838.00 | 2026-04-07 10:15:00 | 20315.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-13 10:15:00 | 22280.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-15 09:15:00 | 22983.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-16 13:15:00 | 22230.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-22 14:00:00 | 21792.00 | 2026-04-23 14:15:00 | 20702.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 14:00:00 | 21792.00 | 2026-04-27 09:15:00 | 20800.00 | STOP_HIT | 0.50 | 4.55% |
