# Honeywell Automation India Ltd. (HONAUT)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 30210.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 66 |
| ALERT1 | 42 |
| ALERT2 | 41 |
| ALERT2_SKIP | 21 |
| ALERT3 | 116 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 46
- **Target hits / Stop hits / Partials:** 1 / 59 / 2
- **Avg / median % per leg:** 0.08% / -0.46%
- **Sum % (uncompounded):** 4.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.15% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -0.15% | -2.5% |
| SELL (all) | 46 | 15 | 32.6% | 0 | 44 | 2 | 0.16% | 7.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 15 | 32.6% | 0 | 44 | 2 | 0.16% | 7.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 16 | 25.8% | 1 | 59 | 2 | 0.08% | 4.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 35095.00 | 34804.52 | 34779.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 35250.00 | 34952.09 | 34854.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 38150.00 | 38360.84 | 37638.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 37990.00 | 38360.84 | 37638.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 37690.00 | 38069.46 | 37824.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 37625.00 | 38069.46 | 37824.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 37725.00 | 38000.57 | 37815.05 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 37070.00 | 37652.31 | 37698.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 37010.00 | 37523.85 | 37635.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 37790.00 | 37577.08 | 37649.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 37790.00 | 37577.08 | 37649.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 37715.00 | 37604.66 | 37655.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 37780.00 | 37604.66 | 37655.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 37625.00 | 37607.99 | 37648.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 37650.00 | 37607.99 | 37648.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 37680.00 | 37622.39 | 37651.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 37680.00 | 37622.39 | 37651.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 37830.00 | 37663.91 | 37667.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 37830.00 | 37663.91 | 37667.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 37785.00 | 37688.13 | 37678.14 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 37475.00 | 37645.88 | 37661.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 37295.00 | 37575.71 | 37627.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 37375.00 | 37359.75 | 37483.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 37375.00 | 37359.75 | 37483.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 37395.00 | 37366.80 | 37475.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 37430.00 | 37366.80 | 37475.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 37210.00 | 37335.44 | 37451.33 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 37500.00 | 37462.84 | 37460.38 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 37205.00 | 37422.02 | 37442.92 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 37735.00 | 37466.75 | 37455.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 37975.00 | 37568.40 | 37502.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 38005.00 | 38105.66 | 37912.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 38005.00 | 38105.66 | 37912.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 37900.00 | 38064.53 | 37911.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 38240.00 | 38064.53 | 37911.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 38235.00 | 38815.53 | 38835.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 38235.00 | 38815.53 | 38835.24 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 39440.00 | 38888.21 | 38830.71 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 09:15:00 | 38990.00 | 39058.13 | 39062.69 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 39120.00 | 39070.51 | 39067.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 38950.00 | 39055.12 | 39061.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 38900.00 | 39011.42 | 39039.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 38980.00 | 38893.76 | 38944.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 38985.00 | 38893.76 | 38944.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 38810.00 | 38877.01 | 38932.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 39015.00 | 38877.01 | 38932.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 38125.00 | 38119.66 | 38377.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 37855.00 | 38087.73 | 38339.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 37995.00 | 38097.18 | 38321.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 38005.00 | 38078.20 | 38272.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 38020.00 | 38097.60 | 38203.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 37925.00 | 38004.43 | 38119.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 38110.00 | 38004.43 | 38119.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 37905.00 | 37984.55 | 38100.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 37780.00 | 37887.09 | 38011.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 37685.00 | 37846.67 | 37981.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 37750.00 | 37796.67 | 37933.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 37795.00 | 37680.10 | 37774.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 37705.00 | 37709.07 | 37772.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:15:00 | 37635.00 | 37709.07 | 37772.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 37875.00 | 37690.30 | 37727.51 | SL hit (close>static) qty=1.00 sl=37835.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 37855.00 | 37763.15 | 37755.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 38430.00 | 37905.85 | 37822.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 38630.00 | 38647.81 | 38465.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:00:00 | 38630.00 | 38647.81 | 38465.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 38925.00 | 39287.98 | 39121.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 38925.00 | 39287.98 | 39121.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 38615.00 | 39153.38 | 39075.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 38560.00 | 39153.38 | 39075.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 38825.00 | 38997.13 | 39013.84 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 40330.00 | 39151.43 | 39039.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 40835.00 | 39488.14 | 39202.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 40700.00 | 40731.38 | 40200.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 13:00:00 | 40700.00 | 40731.38 | 40200.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 40685.00 | 40823.52 | 40715.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 40720.00 | 40823.52 | 40715.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 40650.00 | 40788.82 | 40709.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 40615.00 | 40788.82 | 40709.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 40625.00 | 40756.06 | 40701.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 40625.00 | 40756.06 | 40701.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 40470.00 | 40698.84 | 40680.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 40355.00 | 40698.84 | 40680.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 40485.00 | 40656.08 | 40663.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 40280.00 | 40517.69 | 40590.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 40670.00 | 40516.52 | 40575.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 40670.00 | 40516.52 | 40575.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 40425.00 | 40498.22 | 40561.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 40605.00 | 40498.22 | 40561.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 40635.00 | 40525.57 | 40568.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 40725.00 | 40525.57 | 40568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 40740.00 | 40568.46 | 40583.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 40435.00 | 40568.46 | 40583.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 40505.00 | 40555.77 | 40576.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 40295.00 | 40480.49 | 40536.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 40295.00 | 40423.41 | 40493.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 40600.00 | 40517.22 | 40510.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 40600.00 | 40517.22 | 40510.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 40600.00 | 40517.22 | 40510.63 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 40460.00 | 40505.78 | 40506.03 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 40965.00 | 40597.62 | 40547.75 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 40455.00 | 40597.01 | 40604.61 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 40705.00 | 40616.38 | 40608.42 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 40480.00 | 40592.57 | 40601.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 40300.00 | 40524.84 | 40568.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 40545.00 | 40528.87 | 40566.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 40240.00 | 40505.78 | 40545.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 40575.00 | 40547.30 | 40544.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 40575.00 | 40547.30 | 40544.24 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 40480.00 | 40533.84 | 40538.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 40320.00 | 40491.07 | 40518.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 40425.00 | 40419.30 | 40468.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 40425.00 | 40419.30 | 40468.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 40395.00 | 40414.44 | 40462.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 40430.00 | 40414.44 | 40462.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 40880.00 | 40507.55 | 40500.17 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 40440.00 | 40486.43 | 40491.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 40310.00 | 40451.15 | 40474.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 40540.00 | 40415.63 | 40447.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 40540.00 | 40415.63 | 40447.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 40500.00 | 40432.50 | 40452.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 40590.00 | 40432.50 | 40452.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 40540.00 | 40454.00 | 40460.28 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 40515.00 | 40466.20 | 40465.26 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 40140.00 | 40406.37 | 40438.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 40000.00 | 40325.09 | 40398.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 38375.00 | 38312.56 | 38742.24 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 38790.00 | 38711.57 | 38711.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 15:15:00 | 39235.00 | 38829.67 | 38768.10 | Break + close above crossover candle high |

### Cycle 30 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 37450.00 | 38553.74 | 38648.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 36930.00 | 37418.38 | 37745.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 36885.00 | 36853.98 | 37156.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:30:00 | 36905.00 | 36853.98 | 37156.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 36075.00 | 35885.34 | 36042.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 36075.00 | 35885.34 | 36042.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 36155.00 | 35939.27 | 36053.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 36155.00 | 35939.27 | 36053.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 36775.00 | 36182.53 | 36148.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 37260.00 | 36501.62 | 36306.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 37990.00 | 38155.87 | 37931.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 37990.00 | 38155.87 | 37931.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 37725.00 | 38069.70 | 37912.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 37725.00 | 38069.70 | 37912.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 37785.00 | 38012.76 | 37900.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:15:00 | 37905.00 | 38012.76 | 37900.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 37905.00 | 37965.77 | 37898.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 37905.00 | 38077.88 | 38100.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 37905.00 | 38077.88 | 38100.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 37905.00 | 38077.88 | 38100.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 37875.00 | 38037.30 | 38080.04 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 39075.00 | 38234.87 | 38161.81 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 38120.00 | 38517.74 | 38543.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 37885.00 | 38330.75 | 38450.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 35950.00 | 35871.84 | 36329.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 35950.00 | 35871.84 | 36329.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 35865.00 | 35666.61 | 36033.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 36260.00 | 35666.61 | 36033.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 35660.00 | 35665.29 | 35999.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 35960.00 | 35665.29 | 35999.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 36230.00 | 35778.23 | 36020.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 36200.00 | 35778.23 | 36020.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 36140.00 | 35850.58 | 36031.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 36045.00 | 35883.47 | 36029.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 36505.00 | 35918.05 | 35977.89 | SL hit (close>static) qty=1.00 sl=36265.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 36190.00 | 36019.15 | 36016.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 36360.00 | 36130.66 | 36070.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 36500.00 | 36563.46 | 36374.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 36500.00 | 36563.46 | 36374.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 36890.00 | 37082.10 | 36913.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 36890.00 | 37082.10 | 36913.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 36795.00 | 37024.68 | 36902.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:45:00 | 36705.00 | 37024.68 | 36902.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 36740.00 | 36967.74 | 36888.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 36885.00 | 36967.74 | 36888.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 36915.00 | 36986.96 | 36920.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 36935.00 | 36986.96 | 36920.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 36945.00 | 36978.57 | 36922.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 37225.00 | 36928.07 | 36910.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 37215.00 | 37350.39 | 37351.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 37215.00 | 37350.39 | 37351.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 36985.00 | 37153.32 | 37242.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 36900.00 | 36843.11 | 36987.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 36885.00 | 36843.11 | 36987.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 36795.00 | 36837.79 | 36960.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 36635.00 | 36797.23 | 36930.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 37015.00 | 36746.63 | 36854.77 | SL hit (close>static) qty=1.00 sl=37000.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 36680.00 | 36756.31 | 36849.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:45:00 | 36675.00 | 36696.62 | 36788.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 36605.00 | 35940.29 | 35856.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 36605.00 | 35940.29 | 35856.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 36605.00 | 35940.29 | 35856.17 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 36215.00 | 36265.05 | 36267.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 36045.00 | 36221.04 | 36247.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 36190.00 | 36129.46 | 36185.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 36190.00 | 36129.46 | 36185.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 36080.00 | 36119.57 | 36176.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 36125.00 | 36119.57 | 36176.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 36115.00 | 36118.66 | 36170.75 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 36250.00 | 36152.58 | 36147.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 13:15:00 | 36320.00 | 36203.56 | 36173.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 36270.00 | 36271.04 | 36223.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 36330.00 | 36271.04 | 36223.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 36295.00 | 36275.83 | 36230.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 36295.00 | 36275.83 | 36230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 36270.00 | 36274.66 | 36233.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:15:00 | 36210.00 | 36274.66 | 36233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 36330.00 | 36285.73 | 36242.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 36255.00 | 36285.73 | 36242.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 36350.00 | 36298.59 | 36252.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 36480.00 | 36324.87 | 36268.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 36155.00 | 36291.57 | 36268.27 | SL hit (close<static) qty=1.00 sl=36175.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 36175.00 | 36252.81 | 36253.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 36160.00 | 36234.25 | 36245.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 36315.00 | 36250.40 | 36251.56 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 36300.00 | 36252.25 | 36251.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 36645.00 | 36341.64 | 36293.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 36870.00 | 36893.29 | 36707.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 36870.00 | 36893.29 | 36707.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 36700.00 | 36859.09 | 36771.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 36980.00 | 36896.28 | 36796.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 37100.00 | 37112.63 | 37017.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 37030.00 | 37112.63 | 37017.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 36920.00 | 37074.10 | 37008.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 36920.00 | 37074.10 | 37008.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 36900.00 | 37039.28 | 36998.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 37165.00 | 37039.28 | 36998.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 36690.00 | 36976.74 | 36977.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 36690.00 | 36976.74 | 36977.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 36520.00 | 36835.51 | 36910.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 36915.00 | 36715.31 | 36816.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 37025.00 | 36715.31 | 36816.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 36750.00 | 36722.25 | 36810.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 36700.00 | 36742.24 | 36804.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:30:00 | 36705.00 | 36719.23 | 36783.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 37225.00 | 36818.91 | 36817.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 37225.00 | 36818.91 | 36817.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 37225.00 | 36818.91 | 36817.83 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 36150.00 | 36712.86 | 36788.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 35475.00 | 35833.49 | 36045.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 35640.00 | 35633.93 | 35831.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 35530.00 | 35613.14 | 35803.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 35580.00 | 35583.21 | 35756.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 35530.00 | 35515.21 | 35600.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 35585.00 | 35532.17 | 35600.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 35580.00 | 35541.73 | 35598.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 35565.00 | 35541.73 | 35598.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 35965.00 | 34888.37 | 34769.85 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 35160.00 | 35408.03 | 35433.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 34950.00 | 35098.58 | 35226.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 35265.00 | 35116.89 | 35211.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 35265.00 | 35116.89 | 35211.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 35115.00 | 35116.51 | 35202.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 34950.00 | 35091.21 | 35183.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:00:00 | 34825.00 | 35017.77 | 35132.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 34085.00 | 34047.55 | 34046.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 34085.00 | 34047.55 | 34046.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 34085.00 | 34047.55 | 34046.35 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 33880.00 | 34014.04 | 34031.23 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 34135.00 | 34051.99 | 34046.35 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 33950.00 | 34024.87 | 34034.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 33760.00 | 33950.33 | 33996.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 33860.00 | 33808.24 | 33895.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 33860.00 | 33808.24 | 33895.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 33740.00 | 33794.60 | 33880.96 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 33930.00 | 33795.96 | 33794.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 33990.00 | 33834.77 | 33811.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 33755.00 | 33818.82 | 33806.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 33755.00 | 33818.82 | 33806.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 33860.00 | 33827.05 | 33811.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 33845.00 | 33827.05 | 33811.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 09:15:00 | 33695.00 | 33800.64 | 33800.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 33605.00 | 33723.89 | 33762.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 32995.00 | 32976.78 | 33057.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:45:00 | 33000.00 | 32976.78 | 33057.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 32985.00 | 32982.14 | 33046.25 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 33430.00 | 33065.65 | 33021.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 33665.00 | 33335.99 | 33176.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 34155.00 | 34159.36 | 33852.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 34155.00 | 34159.36 | 33852.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 34090.00 | 34152.46 | 34000.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 34090.00 | 34152.46 | 34000.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 34090.00 | 34139.97 | 34008.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 34220.00 | 34109.01 | 34032.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 33965.00 | 34097.17 | 34041.60 | SL hit (close<static) qty=1.00 sl=34005.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 33795.00 | 33994.79 | 34002.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 33685.00 | 33867.93 | 33936.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 33300.00 | 33256.09 | 33442.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 33100.00 | 33321.30 | 33412.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 33185.00 | 33258.63 | 33365.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 33455.00 | 33414.29 | 33408.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 33455.00 | 33414.29 | 33408.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 33455.00 | 33414.29 | 33408.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 14:15:00 | 34000.00 | 33586.36 | 33499.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 33675.00 | 33710.27 | 33577.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 33705.00 | 33710.27 | 33577.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 33705.00 | 33709.22 | 33588.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 33815.00 | 33730.37 | 33609.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:45:00 | 33790.00 | 33742.30 | 33625.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 33790.00 | 33733.84 | 33632.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 33815.00 | 33737.07 | 33643.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 33850.00 | 33759.66 | 33662.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 33540.00 | 33759.66 | 33662.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 33885.00 | 33784.73 | 33682.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 33545.00 | 33784.73 | 33682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 33635.00 | 33801.26 | 33730.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 33635.00 | 33801.26 | 33730.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | SL hit (close<static) qty=1.00 sl=33575.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | SL hit (close<static) qty=1.00 sl=33575.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | SL hit (close<static) qty=1.00 sl=33575.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 33400.00 | 33721.00 | 33700.71 | SL hit (close<static) qty=1.00 sl=33575.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-20 14:45:00 | 33425.00 | 33721.00 | 33700.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 33915.00 | 33831.64 | 33758.97 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 33500.00 | 33718.60 | 33722.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 33295.00 | 33545.47 | 33630.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 31155.00 | 31080.75 | 31604.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:45:00 | 31405.00 | 31080.75 | 31604.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 31470.00 | 31158.60 | 31592.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 31570.00 | 31158.60 | 31592.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 31545.00 | 31290.50 | 31580.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 31545.00 | 31290.50 | 31580.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 31475.00 | 31327.40 | 31570.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 31445.00 | 31488.58 | 31582.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 32100.00 | 31610.87 | 31629.61 | SL hit (close>static) qty=1.00 sl=31600.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 32100.00 | 31708.69 | 31672.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 32300.00 | 31888.76 | 31764.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 33070.00 | 33178.10 | 32823.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 33070.00 | 33178.10 | 32823.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 32850.00 | 33112.48 | 32825.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 32850.00 | 33112.48 | 32825.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 33195.00 | 33128.98 | 32859.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 32580.00 | 33128.98 | 32859.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 32605.00 | 33024.19 | 32836.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:45:00 | 33375.00 | 32975.28 | 32846.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 33200.00 | 33457.60 | 33358.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 32985.00 | 33296.26 | 33298.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 32985.00 | 33296.26 | 33298.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 32985.00 | 33296.26 | 33298.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 32710.00 | 33131.61 | 33220.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 32865.00 | 32755.05 | 32929.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 32865.00 | 32755.05 | 32929.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 32415.00 | 32687.04 | 32883.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 32300.00 | 32609.63 | 32830.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 32300.00 | 32460.74 | 32577.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 32105.00 | 32440.90 | 32537.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 32270.00 | 32276.42 | 32382.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 32155.00 | 32252.14 | 32361.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 32335.00 | 32252.14 | 32361.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 31370.00 | 31320.79 | 31465.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 31540.00 | 31320.79 | 31465.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 31435.00 | 31343.63 | 31462.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 31495.00 | 31343.63 | 31462.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 31360.00 | 31365.33 | 31436.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 31300.00 | 31368.06 | 31404.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 31255.00 | 31341.56 | 31386.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 31370.00 | 31185.08 | 31176.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 31560.00 | 31300.85 | 31232.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 31430.00 | 31557.88 | 31455.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 31390.00 | 31557.88 | 31455.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 31380.00 | 31522.31 | 31448.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:30:00 | 31310.00 | 31522.31 | 31448.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 31375.00 | 31454.88 | 31428.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 31345.00 | 31454.88 | 31428.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 31355.00 | 31432.52 | 31422.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 31305.00 | 31432.52 | 31422.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 09:15:00 | 31265.00 | 31399.02 | 31408.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 31130.00 | 31270.70 | 31328.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 30365.00 | 30211.80 | 30482.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 30365.00 | 30211.80 | 30482.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 30200.00 | 30156.35 | 30406.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 30015.00 | 30144.08 | 30378.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 30000.00 | 30115.27 | 30343.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 30025.00 | 30108.21 | 30319.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 29985.00 | 30079.57 | 30287.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 30535.00 | 29875.52 | 29932.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 30405.00 | 29875.52 | 29932.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 30495.00 | 29999.41 | 29983.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 30645.00 | 30270.46 | 30124.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 30720.00 | 30721.94 | 30477.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:45:00 | 30725.00 | 30721.94 | 30477.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 30630.00 | 30660.92 | 30506.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 30690.00 | 30666.73 | 30523.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 30780.00 | 30609.93 | 30529.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 30285.00 | 30471.02 | 30488.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 30285.00 | 30471.02 | 30488.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 30285.00 | 30471.02 | 30488.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 30120.00 | 30400.81 | 30455.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 29500.00 | 29479.09 | 29784.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:00:00 | 29500.00 | 29479.09 | 29784.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 29480.00 | 29403.25 | 29606.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 29530.00 | 29403.25 | 29606.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 29470.00 | 29427.50 | 29568.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 29470.00 | 29427.50 | 29568.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 29510.00 | 29444.00 | 29562.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:30:00 | 29575.00 | 29444.00 | 29562.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 28920.00 | 28847.31 | 29074.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 28895.00 | 28847.31 | 29074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 27840.00 | 27528.70 | 27896.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 27840.00 | 27528.70 | 27896.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 27745.00 | 27571.96 | 27882.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 27860.00 | 27571.96 | 27882.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 28255.00 | 27717.76 | 27872.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 28205.00 | 27717.76 | 27872.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 28220.00 | 27818.21 | 27904.42 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 28220.00 | 27958.85 | 27957.12 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 27655.00 | 27923.92 | 27953.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 27420.00 | 27771.31 | 27876.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 27595.00 | 26843.12 | 27139.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 27595.00 | 26843.12 | 27139.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 27445.00 | 26963.50 | 27167.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 27295.00 | 27089.80 | 27206.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 27300.00 | 27231.67 | 27255.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 27300.00 | 27231.67 | 27255.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 27360.00 | 27082.43 | 27063.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 27995.00 | 27334.03 | 27224.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 27510.00 | 27582.63 | 27399.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 13:45:00 | 27525.00 | 27582.63 | 27399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 27090.00 | 27484.10 | 27371.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 27090.00 | 27484.10 | 27371.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 27500.00 | 27487.28 | 27383.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 27965.00 | 27487.28 | 27383.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 30761.50 | 30202.76 | 29470.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 31830.00 | 32388.51 | 32451.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 31650.00 | 32240.81 | 32378.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 31825.00 | 31731.79 | 31983.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 31825.00 | 31731.79 | 31983.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 31780.00 | 31782.46 | 31931.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 31600.00 | 31745.97 | 31901.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 31640.00 | 31724.78 | 31877.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:15:00 | 30020.00 | 30522.71 | 30803.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:15:00 | 30058.00 | 30522.71 | 30803.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 30445.00 | 30410.71 | 30673.70 | SL hit (close>ema200) qty=0.50 sl=30410.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 30445.00 | 30410.71 | 30673.70 | SL hit (close>ema200) qty=0.50 sl=30410.71 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:30:00 | 34935.00 | 2025-05-12 12:15:00 | 35095.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-29 09:15:00 | 38240.00 | 2025-06-04 09:15:00 | 38235.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-06-16 09:30:00 | 37855.00 | 2025-06-23 10:15:00 | 37875.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-06-16 11:15:00 | 37995.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-06-16 13:15:00 | 38005.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-06-17 12:00:00 | 38020.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-06-18 14:15:00 | 37780.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-06-18 15:00:00 | 37685.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-19 10:00:00 | 37750.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-06-20 10:45:00 | 37795.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-06-20 13:15:00 | 37635.00 | 2025-06-23 13:15:00 | 37855.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-11 12:15:00 | 40295.00 | 2025-07-14 15:15:00 | 40600.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-11 15:15:00 | 40295.00 | 2025-07-14 15:15:00 | 40600.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-18 15:15:00 | 40240.00 | 2025-07-21 14:15:00 | 40575.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-21 14:15:00 | 37905.00 | 2025-08-26 13:15:00 | 37905.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-08-22 09:15:00 | 37905.00 | 2025-08-26 13:15:00 | 37905.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-09 11:45:00 | 36045.00 | 2025-09-10 09:15:00 | 36505.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-17 09:15:00 | 37225.00 | 2025-09-22 09:15:00 | 37215.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-09-24 13:00:00 | 36635.00 | 2025-09-25 09:15:00 | 37015.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-25 11:15:00 | 36680.00 | 2025-10-08 09:15:00 | 36605.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-09-25 14:45:00 | 36675.00 | 2025-10-08 09:15:00 | 36605.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-10-21 13:45:00 | 36480.00 | 2025-10-23 10:15:00 | 36155.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-31 09:15:00 | 37165.00 | 2025-10-31 10:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-03 12:30:00 | 36700.00 | 2025-11-04 09:15:00 | 37225.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-03 14:30:00 | 36705.00 | 2025-11-04 09:15:00 | 37225.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-12 12:00:00 | 35530.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-11-12 13:45:00 | 35580.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-14 10:15:00 | 35530.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-11-14 11:15:00 | 35585.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-14 12:15:00 | 35565.00 | 2025-11-24 14:15:00 | 35965.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-04 12:45:00 | 34950.00 | 2025-12-15 15:15:00 | 34085.00 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-12-04 15:00:00 | 34825.00 | 2025-12-15 15:15:00 | 34085.00 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2026-01-08 09:15:00 | 34220.00 | 2026-01-08 10:15:00 | 33965.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-13 15:15:00 | 33100.00 | 2026-01-14 15:15:00 | 33455.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-14 10:00:00 | 33185.00 | 2026-01-14 15:15:00 | 33455.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-19 12:00:00 | 33815.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-19 12:45:00 | 33790.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-19 13:45:00 | 33790.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-19 14:45:00 | 33815.00 | 2026-01-20 14:15:00 | 33400.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-29 10:30:00 | 31445.00 | 2026-01-29 11:15:00 | 32100.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-02-02 13:45:00 | 33375.00 | 2026-02-04 14:15:00 | 32985.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-04 13:00:00 | 33200.00 | 2026-02-04 14:15:00 | 32985.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-06 13:00:00 | 32300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.88% |
| SELL | retest2 | 2026-02-09 15:15:00 | 32300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.88% |
| SELL | retest2 | 2026-02-10 11:15:00 | 32105.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2026-02-11 11:15:00 | 32270.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | 2.79% |
| SELL | retest2 | 2026-02-19 09:45:00 | 31300.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-02-19 11:45:00 | 31255.00 | 2026-02-23 12:15:00 | 31370.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-03-05 11:15:00 | 30015.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-05 12:00:00 | 30000.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-03-05 12:30:00 | 30025.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-03-05 13:30:00 | 29985.00 | 2026-03-10 10:15:00 | 30495.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-03-12 11:00:00 | 30690.00 | 2026-03-13 11:15:00 | 30285.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-03-12 14:15:00 | 30780.00 | 2026-03-13 11:15:00 | 30285.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-01 11:30:00 | 27295.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-04-01 13:45:00 | 27300.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-04-01 14:15:00 | 27300.00 | 2026-04-06 13:15:00 | 27360.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-04-09 09:15:00 | 27965.00 | 2026-04-16 09:15:00 | 30761.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:00:00 | 31600.00 | 2026-05-06 13:15:00 | 30020.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 12:00:00 | 31640.00 | 2026-05-06 13:15:00 | 30058.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 11:00:00 | 31600.00 | 2026-05-07 09:15:00 | 30445.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2026-04-28 12:00:00 | 31640.00 | 2026-05-07 09:15:00 | 30445.00 | STOP_HIT | 0.50 | 3.78% |
