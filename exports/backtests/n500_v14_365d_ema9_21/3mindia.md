# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 86 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 20 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 75 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 52
- **Target hits / Stop hits / Partials:** 0 / 83 / 8
- **Avg / median % per leg:** 0.62% / -0.50%
- **Sum % (uncompounded):** 56.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 16 | 36.4% | 0 | 44 | 0 | 0.09% | 4.1% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.07% | -6.4% |
| BUY @ 3rd Alert (retest2) | 38 | 15 | 39.5% | 0 | 38 | 0 | 0.28% | 10.5% |
| SELL (all) | 47 | 23 | 48.9% | 0 | 39 | 8 | 1.12% | 52.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.13% | -2.3% |
| SELL @ 3rd Alert (retest2) | 45 | 23 | 51.1% | 0 | 37 | 8 | 1.21% | 54.7% |
| retest1 (combined) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.08% | -8.6% |
| retest2 (combined) | 83 | 38 | 45.8% | 0 | 75 | 8 | 0.79% | 65.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 29815.00 | 29354.77 | 29347.46 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 29100.00 | 29342.14 | 29366.11 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 29505.00 | 29392.77 | 29386.08 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 29130.00 | 29340.22 | 29364.28 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 29585.00 | 29379.78 | 29360.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 29735.00 | 29457.26 | 29399.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 29565.00 | 29637.09 | 29570.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 29650.00 | 29639.67 | 29578.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:30:00 | 29580.00 | 29639.67 | 29578.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 29580.00 | 29621.39 | 29580.04 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 29400.00 | 29567.37 | 29570.88 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 29650.00 | 29568.81 | 29567.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 29945.00 | 29665.51 | 29613.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 29860.00 | 29914.81 | 29793.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:30:00 | 29860.00 | 29914.81 | 29793.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 30055.00 | 30137.05 | 30021.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 30135.00 | 30137.05 | 30021.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 30130.00 | 30197.01 | 30099.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 30075.00 | 30197.01 | 30099.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 30075.00 | 30172.61 | 30097.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 30080.00 | 30172.61 | 30097.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 30215.00 | 30181.09 | 30108.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 30080.00 | 30181.09 | 30108.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 30145.00 | 30173.87 | 30111.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 29980.00 | 30173.87 | 30111.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 30090.00 | 30157.09 | 30109.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 29995.00 | 30157.09 | 30109.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 30135.00 | 30152.68 | 30111.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 30270.00 | 30152.68 | 30111.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 29990.00 | 30125.93 | 30111.29 | SL hit (close<static) qty=1.00 sl=30005.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 30020.00 | 30089.40 | 30096.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 29960.00 | 30063.52 | 30083.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 29000.00 | 28998.53 | 29297.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:30:00 | 29015.00 | 28998.53 | 29297.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 29335.00 | 29082.06 | 29284.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 29350.00 | 29082.06 | 29284.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 29315.00 | 29128.65 | 29287.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 29230.00 | 29161.92 | 29288.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:30:00 | 29270.00 | 29200.15 | 29266.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 29265.00 | 29265.63 | 29281.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 29620.00 | 29383.60 | 29334.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 13:15:00 | 29815.00 | 29833.42 | 29669.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 29815.00 | 29833.42 | 29669.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 29675.00 | 29801.73 | 29670.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 29660.00 | 29801.73 | 29670.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 29660.00 | 29773.39 | 29669.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 29735.00 | 29773.39 | 29669.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 29755.00 | 29769.71 | 29676.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 29765.00 | 29771.77 | 29686.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 29605.00 | 29723.02 | 29684.56 | SL hit (close<static) qty=1.00 sl=29640.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 29605.00 | 29723.02 | 29684.56 | SL hit (close<static) qty=1.00 sl=29640.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 29605.00 | 29723.02 | 29684.56 | SL hit (close<static) qty=1.00 sl=29640.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 29550.00 | 29662.75 | 29664.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 29500.00 | 29630.20 | 29649.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 29375.00 | 29440.18 | 29518.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 29450.00 | 29433.66 | 29469.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 29420.00 | 29430.93 | 29464.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 29355.00 | 29439.00 | 29459.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 29355.00 | 29422.20 | 29450.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 29515.00 | 29422.20 | 29450.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 29495.00 | 29436.76 | 29454.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 29355.00 | 29420.41 | 29445.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:30:00 | 29370.00 | 29432.32 | 29448.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 29395.00 | 29432.32 | 29448.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 10:15:00 | 28775.00 | 28663.79 | 28617.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 28570.00 | 28645.03 | 28612.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 28465.00 | 28609.02 | 28599.38 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 28500.00 | 28587.22 | 28590.35 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 28645.00 | 28596.21 | 28591.08 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 28445.00 | 28580.07 | 28593.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 28400.00 | 28503.40 | 28551.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 28785.00 | 28529.94 | 28548.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 28670.00 | 28557.95 | 28559.83 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 28770.00 | 28600.36 | 28578.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 29305.00 | 28783.17 | 28678.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 28900.00 | 28965.43 | 28835.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:15:00 | 29370.00 | 28965.43 | 28835.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 29565.00 | 29536.95 | 29427.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 29670.00 | 29555.56 | 29445.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 29670.00 | 29702.36 | 29592.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 29600.00 | 29689.58 | 29635.40 | SL hit (close<ema400) qty=1.00 sl=29635.40 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 29670.00 | 29660.26 | 29632.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 29705.00 | 29660.26 | 29632.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 29680.00 | 29664.21 | 29637.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 29680.00 | 29664.21 | 29637.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 29670.00 | 29665.37 | 29640.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 29670.00 | 29665.37 | 29640.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 29700.00 | 29672.30 | 29645.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 30100.00 | 29672.30 | 29645.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:15:00 | 29765.00 | 29773.66 | 29711.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 29905.00 | 29753.95 | 29716.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 29550.00 | 29713.16 | 29701.42 | SL hit (close<static) qty=1.00 sl=29610.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 29550.00 | 29713.16 | 29701.42 | SL hit (close<static) qty=1.00 sl=29610.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 29550.00 | 29713.16 | 29701.42 | SL hit (close<static) qty=1.00 sl=29610.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 29840.00 | 29707.53 | 29699.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 29860.00 | 29969.35 | 29923.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 29855.00 | 29969.35 | 29923.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 29830.00 | 29941.48 | 29915.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 29905.00 | 29941.48 | 29915.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 29965.00 | 29948.56 | 29923.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 15:15:00 | 31315.00 | 31221.98 | 31216.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 31575.00 | 31337.07 | 31272.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 31165.00 | 31302.65 | 31262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 31235.00 | 31289.12 | 31260.23 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 31030.00 | 31237.30 | 31239.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 14:15:00 | 30875.00 | 31164.84 | 31206.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 31065.00 | 30890.37 | 31002.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 30960.00 | 30904.29 | 30999.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 31155.00 | 30904.29 | 30999.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 31030.00 | 30929.44 | 31001.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 31030.00 | 30929.44 | 31001.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 31305.00 | 31004.55 | 31029.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 31360.00 | 31004.55 | 31029.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 31130.00 | 31049.71 | 31046.86 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 30930.00 | 31025.77 | 31036.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 30850.00 | 30990.62 | 31019.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 30845.00 | 30819.91 | 30914.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 13:00:00 | 30845.00 | 30819.91 | 30914.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 31070.00 | 30869.93 | 30928.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 31040.00 | 30869.93 | 30928.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 30805.00 | 30856.94 | 30917.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:15:00 | 31050.00 | 30856.94 | 30917.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 31050.00 | 30895.55 | 30929.17 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 31465.00 | 31009.44 | 30977.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 31710.00 | 31149.55 | 31044.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 30965.00 | 31275.15 | 31158.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 30800.00 | 31180.12 | 31125.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 30610.00 | 31180.12 | 31125.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 30610.00 | 31066.10 | 31078.73 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 30950.00 | 30801.34 | 30787.19 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 30460.00 | 30733.07 | 30757.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 30215.00 | 30629.46 | 30708.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 30595.00 | 30558.65 | 30658.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 30925.00 | 30631.92 | 30682.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 30925.00 | 30631.92 | 30682.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 32000.00 | 30905.54 | 30802.34 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 30770.00 | 31163.13 | 31214.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 30610.00 | 31052.50 | 31159.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 13:15:00 | 31020.00 | 30996.80 | 31112.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 13:45:00 | 31000.00 | 30996.80 | 31112.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 30665.00 | 30937.12 | 31057.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:45:00 | 30595.00 | 30877.70 | 31019.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 30450.00 | 30747.13 | 30902.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 30515.00 | 30432.83 | 30568.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 31080.00 | 30702.56 | 30618.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 31000.00 | 31034.69 | 30847.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 31000.00 | 31034.69 | 30847.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 30950.00 | 31010.60 | 30868.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 30950.00 | 31010.60 | 30868.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 30945.00 | 30997.48 | 30875.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 30850.00 | 30997.48 | 30875.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 30905.00 | 30978.98 | 30878.45 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 30625.00 | 30850.36 | 30862.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 30425.00 | 30765.29 | 30823.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 30655.00 | 30627.10 | 30719.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:15:00 | 30275.00 | 30627.10 | 30719.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:30:00 | 30500.00 | 30590.14 | 30685.70 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 30730.00 | 30621.27 | 30676.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 30730.00 | 30621.27 | 30676.28 | SL hit (close>ema400) qty=1.00 sl=30676.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 30730.00 | 30621.27 | 30676.28 | SL hit (close>ema400) qty=1.00 sl=30676.28 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 30790.00 | 30621.27 | 30676.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 30505.00 | 30598.02 | 30660.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 30265.00 | 30604.42 | 30657.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 30455.00 | 30559.23 | 30626.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 30400.00 | 30543.38 | 30613.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 30400.00 | 30503.81 | 30576.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30468.24 | 30547.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 30940.00 | 30665.14 | 30597.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 31140.00 | 31251.08 | 31060.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 31140.00 | 31228.86 | 31068.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 31260.00 | 31228.86 | 31068.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 30990.00 | 31226.66 | 31148.35 | SL hit (close<static) qty=1.00 sl=31050.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 30755.00 | 31092.86 | 31098.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 30380.00 | 30664.40 | 30849.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 30910.00 | 30524.00 | 30637.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 30845.00 | 30588.20 | 30656.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 30845.00 | 30588.20 | 30656.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 30845.00 | 30721.54 | 30705.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 31300.00 | 30865.79 | 30775.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 30860.00 | 30997.28 | 30879.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 30845.00 | 30966.83 | 30876.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 30995.00 | 30938.57 | 30877.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 31005.00 | 30906.85 | 30868.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 30780.00 | 30847.03 | 30847.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 30780.00 | 30847.03 | 30847.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 30780.00 | 30847.03 | 30847.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 14:15:00 | 30665.00 | 30810.62 | 30831.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 31020.00 | 30855.04 | 30844.35 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 30700.00 | 30862.66 | 30878.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 30535.00 | 30775.00 | 30829.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 30640.00 | 30591.51 | 30681.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:15:00 | 30700.00 | 30591.51 | 30681.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 30500.00 | 30573.21 | 30664.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 30405.00 | 30523.05 | 30625.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 30400.00 | 30379.57 | 30503.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 30400.00 | 30327.93 | 30425.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28884.75 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28880.00 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28880.00 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 12:15:00 | 29230.00 | 29200.93 | 29397.03 | SL hit (close>ema200) qty=0.50 sl=29200.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 12:15:00 | 29230.00 | 29200.93 | 29397.03 | SL hit (close>ema200) qty=0.50 sl=29200.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 12:15:00 | 29230.00 | 29200.93 | 29397.03 | SL hit (close>ema200) qty=0.50 sl=29200.93 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 29360.00 | 29183.16 | 29171.98 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 29115.00 | 29158.62 | 29162.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 13:15:00 | 29060.00 | 29138.81 | 29152.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 29095.00 | 29069.76 | 29104.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 29095.00 | 29069.76 | 29104.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 29105.00 | 29076.81 | 29104.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:15:00 | 29130.00 | 29076.81 | 29104.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 29130.00 | 29087.45 | 29106.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 29175.00 | 29087.45 | 29106.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 29255.00 | 29120.96 | 29120.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 29315.00 | 29159.77 | 29138.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 29205.00 | 29221.01 | 29181.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 29300.00 | 29221.01 | 29181.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 29180.00 | 29212.81 | 29181.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 29180.00 | 29212.81 | 29181.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 29200.00 | 29210.24 | 29183.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 29150.00 | 29210.24 | 29183.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 29085.00 | 29185.20 | 29174.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 29110.00 | 29185.20 | 29174.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 29100.00 | 29168.16 | 29167.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 29085.00 | 29168.16 | 29167.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 29080.00 | 29150.53 | 29159.75 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 29200.00 | 29166.05 | 29161.51 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 29030.00 | 29151.07 | 29157.57 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 29215.00 | 29157.41 | 29154.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 29335.00 | 29211.74 | 29181.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 29350.00 | 29418.57 | 29324.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 29350.00 | 29418.57 | 29324.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 29350.00 | 29404.85 | 29326.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 29360.00 | 29404.85 | 29326.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 29170.00 | 29357.88 | 29312.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 29170.00 | 29357.88 | 29312.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 28975.00 | 29281.31 | 29282.01 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 29275.00 | 29208.20 | 29202.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 29390.00 | 29244.56 | 29219.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 29200.00 | 29235.65 | 29218.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 29365.00 | 29261.52 | 29231.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 29380.00 | 29288.17 | 29249.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 29380.00 | 29288.17 | 29249.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 29380.00 | 29306.54 | 29261.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 29455.00 | 29384.79 | 29310.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 29625.00 | 29511.42 | 29418.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 29540.00 | 29511.42 | 29418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 29620.00 | 29690.24 | 29618.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 29620.00 | 29690.24 | 29618.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 29595.00 | 29671.19 | 29616.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 29605.00 | 29671.19 | 29616.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 29620.00 | 29660.95 | 29616.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 29555.00 | 29660.95 | 29616.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 29600.00 | 29648.76 | 29615.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 29625.00 | 29648.76 | 29615.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 29645.00 | 29648.01 | 29617.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 29520.00 | 29648.01 | 29617.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 29595.00 | 29637.41 | 29615.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 29725.00 | 29643.99 | 29623.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 10:15:00 | 29440.00 | 29659.46 | 29727.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 29680.00 | 29608.05 | 29681.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 29680.00 | 29608.05 | 29681.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 29775.00 | 29641.44 | 29690.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 29775.00 | 29641.44 | 29690.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 29675.00 | 29648.15 | 29688.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 29670.00 | 29648.15 | 29688.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 29735.00 | 29665.52 | 29692.95 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 29785.00 | 29714.93 | 29712.23 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 29465.00 | 29670.56 | 29692.88 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 29900.00 | 29716.45 | 29711.71 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 29570.00 | 29699.73 | 29705.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 29400.00 | 29639.78 | 29677.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 29595.00 | 29565.78 | 29618.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 29700.00 | 29565.78 | 29618.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 49 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 30550.00 | 29762.63 | 29703.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 30675.00 | 29945.10 | 29791.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 35235.00 | 35550.89 | 34452.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 36200.00 | 35604.14 | 34989.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 12:45:00 | 36025.00 | 35805.72 | 35245.73 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 14:00:00 | 36130.00 | 35870.58 | 35326.12 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 13:15:00 | 36055.00 | 35839.91 | 35563.13 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | SL hit (close<ema400) qty=1.00 sl=35641.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | SL hit (close<ema400) qty=1.00 sl=35641.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | SL hit (close<ema400) qty=1.00 sl=35641.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | SL hit (close<ema400) qty=1.00 sl=35641.27 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 36150.00 | 35931.73 | 35716.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 36155.00 | 36011.88 | 35810.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 36350.00 | 36017.01 | 35847.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 36185.00 | 36050.60 | 35878.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 36025.00 | 36066.19 | 35916.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 36060.00 | 36066.19 | 35916.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 36050.00 | 36063.64 | 35974.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 36355.00 | 35986.22 | 35965.43 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 35920.00 | 36054.59 | 36061.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 11:15:00 | 35890.00 | 36021.67 | 36046.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 15:15:00 | 36000.00 | 35980.49 | 36013.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 35930.00 | 35980.49 | 36013.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 35780.00 | 35940.39 | 35992.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 35690.00 | 35898.31 | 35968.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 35665.00 | 35898.31 | 35968.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 35640.00 | 35781.50 | 35882.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33905.50 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33881.75 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33858.00 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 34500.00 | 34376.86 | 34514.98 | SL hit (close>ema200) qty=0.50 sl=34376.86 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 34500.00 | 34376.86 | 34514.98 | SL hit (close>ema200) qty=0.50 sl=34376.86 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 34500.00 | 34376.86 | 34514.98 | SL hit (close>ema200) qty=0.50 sl=34376.86 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 14:15:00 | 34895.00 | 34634.09 | 34603.95 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 34265.00 | 34541.34 | 34570.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 34055.00 | 34349.54 | 34430.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 34365.00 | 34352.63 | 34424.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 34365.00 | 34352.63 | 34424.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 34365.00 | 34355.10 | 34419.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 34500.00 | 34355.10 | 34419.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 34510.00 | 34386.08 | 34427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 34510.00 | 34386.08 | 34427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 34615.00 | 34431.87 | 34444.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 34615.00 | 34431.87 | 34444.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 34405.00 | 34426.49 | 34441.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 34655.00 | 34426.49 | 34441.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 35340.00 | 34609.19 | 34522.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 35540.00 | 34795.36 | 34615.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 35375.00 | 35484.21 | 35230.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 35265.00 | 35484.21 | 35230.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 35330.00 | 35453.37 | 35239.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 35460.00 | 35440.56 | 35269.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 35565.00 | 35408.02 | 35310.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 35460.00 | 35439.53 | 35342.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 35065.00 | 35308.74 | 35306.78 | SL hit (close<static) qty=1.00 sl=35155.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 35065.00 | 35308.74 | 35306.78 | SL hit (close<static) qty=1.00 sl=35155.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 35065.00 | 35308.74 | 35306.78 | SL hit (close<static) qty=1.00 sl=35155.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 35135.00 | 35274.00 | 35291.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 35005.00 | 35140.52 | 35214.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 35070.00 | 35008.10 | 35094.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:30:00 | 35035.00 | 35008.10 | 35094.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 35205.00 | 35047.48 | 35104.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 35205.00 | 35047.48 | 35104.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 34785.00 | 34994.98 | 35075.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 35110.00 | 34994.98 | 35075.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 35100.00 | 35015.98 | 35077.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 35250.00 | 35015.98 | 35077.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 35290.00 | 35070.79 | 35096.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 35260.00 | 35070.79 | 35096.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 35420.00 | 35140.63 | 35126.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 35535.00 | 35219.50 | 35163.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 35415.00 | 35415.07 | 35312.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 35415.00 | 35415.07 | 35312.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 35225.00 | 35377.06 | 35304.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 35210.00 | 35377.06 | 35304.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 35155.00 | 35332.65 | 35290.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 35165.00 | 35332.65 | 35290.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 35090.00 | 35234.09 | 35249.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 34825.00 | 35050.87 | 35151.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 34395.00 | 34340.85 | 34550.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 34405.00 | 34340.85 | 34550.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 34535.00 | 34379.68 | 34549.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 34705.00 | 34379.68 | 34549.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 34740.00 | 34451.74 | 34566.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 34715.00 | 34451.74 | 34566.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 34820.00 | 34525.40 | 34589.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 34820.00 | 34525.40 | 34589.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 35620.00 | 34807.45 | 34710.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 35970.00 | 35239.15 | 35044.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 35345.00 | 35878.12 | 35912.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 35000.00 | 35702.50 | 35829.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 35065.00 | 34936.55 | 35232.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 35065.00 | 34936.55 | 35232.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 35235.00 | 34996.24 | 35232.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 35235.00 | 34996.24 | 35232.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 34830.00 | 34962.99 | 35196.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 34750.00 | 34942.57 | 35130.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:00:00 | 34590.00 | 34724.82 | 34933.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 34590.00 | 34308.33 | 34483.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 35215.00 | 34876.31 | 34707.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 35065.00 | 35156.03 | 34916.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:45:00 | 35020.00 | 35156.03 | 34916.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 34880.00 | 35100.83 | 34913.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 34880.00 | 35100.83 | 34913.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 34855.00 | 35051.66 | 34908.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 34825.00 | 35051.66 | 34908.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 34835.00 | 35008.33 | 34901.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 34840.00 | 35008.33 | 34901.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 35070.00 | 35117.46 | 34986.13 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 34410.00 | 34887.05 | 34917.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 34160.00 | 34665.31 | 34807.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 34080.00 | 34017.24 | 34299.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 34080.00 | 34017.24 | 34299.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 33990.00 | 34001.06 | 34184.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 34205.00 | 34001.06 | 34184.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 33730.00 | 33896.24 | 34033.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 33645.00 | 33773.60 | 33952.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 33840.00 | 33476.84 | 33472.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 33840.00 | 33476.84 | 33472.61 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 33345.00 | 33455.22 | 33467.81 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 34710.00 | 33713.74 | 33580.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 35670.00 | 35210.81 | 34925.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 35260.00 | 35274.92 | 35007.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 14:15:00 | 35775.00 | 35491.55 | 35202.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 35180.00 | 35456.99 | 35238.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 35180.00 | 35456.99 | 35238.95 | SL hit (close<ema400) qty=1.00 sl=35238.95 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 35075.00 | 35456.99 | 35238.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 35265.00 | 35418.59 | 35241.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 35480.00 | 35228.29 | 35209.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:00:00 | 35425.00 | 35267.63 | 35229.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 35410.00 | 35296.11 | 35245.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 35520.00 | 35367.74 | 35297.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 37560.00 | 37508.57 | 37168.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 37500.00 | 37508.57 | 37168.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 35900.00 | 37152.71 | 37089.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 35855.00 | 37152.71 | 37089.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 35005.00 | 36179.02 | 36587.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 35805.00 | 35476.21 | 36020.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 35805.00 | 35476.21 | 36020.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 36035.00 | 35587.96 | 36022.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 35685.00 | 35923.35 | 36064.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 36125.00 | 36023.43 | 36046.88 | SL hit (close>static) qty=1.00 sl=36100.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 37000.00 | 36232.00 | 36136.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 37700.00 | 36640.48 | 36346.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 36905.00 | 36970.14 | 36657.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 36905.00 | 36970.14 | 36657.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 36945.00 | 36965.11 | 36683.35 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 36325.00 | 36523.00 | 36546.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 35990.00 | 36416.40 | 36496.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 36015.00 | 36176.88 | 36278.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 35700.00 | 36109.50 | 36238.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 35865.00 | 36057.12 | 36172.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 36675.00 | 36182.62 | 36188.36 | SL hit (close>static) qty=1.00 sl=36425.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 36675.00 | 36182.62 | 36188.36 | SL hit (close>static) qty=1.00 sl=36425.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 36675.00 | 36182.62 | 36188.36 | SL hit (close>static) qty=1.00 sl=36425.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 36710.00 | 36288.10 | 36235.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 11:15:00 | 36945.00 | 36501.42 | 36395.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 14:15:00 | 35800.00 | 36518.92 | 36562.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 35180.00 | 36168.11 | 36390.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 34585.00 | 34449.49 | 35003.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 34585.00 | 34449.49 | 35003.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 34145.00 | 33763.73 | 33949.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 34145.00 | 33763.73 | 33949.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 34000.00 | 33810.98 | 33954.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 34270.00 | 33810.98 | 33954.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 33950.00 | 33838.78 | 33953.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 34175.00 | 33838.78 | 33953.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 34220.00 | 33915.03 | 33977.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 34420.00 | 33915.03 | 33977.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 34505.00 | 34033.02 | 34025.84 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 33670.00 | 34048.13 | 34051.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 33250.00 | 33836.01 | 33950.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 34160.00 | 33785.92 | 33875.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 34295.00 | 33887.74 | 33914.00 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 34250.00 | 33960.19 | 33944.54 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 33720.00 | 33912.15 | 33924.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 33305.00 | 33790.72 | 33867.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 32910.00 | 32798.06 | 33141.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 32910.00 | 32798.06 | 33141.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 33410.00 | 32952.76 | 33154.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 33410.00 | 32952.76 | 33154.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 33545.00 | 33071.21 | 33189.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 33230.00 | 33234.70 | 33247.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:45:00 | 33250.00 | 33177.41 | 33214.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 31568.50 | 32298.63 | 32604.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 31587.50 | 32298.63 | 32604.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 31610.00 | 31319.31 | 31739.13 | SL hit (close>ema200) qty=0.50 sl=31319.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 31610.00 | 31319.31 | 31739.13 | SL hit (close>ema200) qty=0.50 sl=31319.31 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 31900.00 | 31799.61 | 31794.98 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 31245.00 | 31688.69 | 31744.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 31185.00 | 31433.26 | 31591.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:30:00 | 30150.00 | 30417.45 | 30741.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 29955.00 | 30417.45 | 30741.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 30260.00 | 29584.37 | 29613.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 31555.00 | 30652.18 | 30190.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 10:15:00 | 31385.00 | 31433.82 | 31171.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 14:15:00 | 31300.00 | 31377.21 | 31225.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 31300.00 | 31377.21 | 31225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 31300.00 | 31377.21 | 31225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 31400.00 | 31381.77 | 31241.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 31630.00 | 31381.77 | 31241.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 31435.00 | 31408.93 | 31387.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:00:00 | 31490.00 | 31425.15 | 31396.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 31440.00 | 31388.03 | 31384.84 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 31210.00 | 31352.43 | 31368.95 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 12:15:00 | 31620.00 | 31420.60 | 31397.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 31985.00 | 31652.37 | 31553.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 32670.00 | 32813.92 | 32463.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 32670.00 | 32813.92 | 32463.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 32310.00 | 32713.13 | 32449.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 32310.00 | 32713.13 | 32449.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 32470.00 | 32664.51 | 32451.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 32325.00 | 32664.51 | 32451.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 32650.00 | 32661.61 | 32469.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 32570.00 | 32661.61 | 32469.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 33005.00 | 33310.86 | 33037.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 33005.00 | 33310.86 | 33037.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 33340.00 | 33316.69 | 33065.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 33350.00 | 33309.35 | 33084.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 33420.00 | 33331.48 | 33115.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 32810.00 | 33228.33 | 33139.82 | SL hit (close<static) qty=1.00 sl=33000.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 32810.00 | 33228.33 | 33139.82 | SL hit (close<static) qty=1.00 sl=33000.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 32885.00 | 33098.07 | 33099.07 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 33250.00 | 33109.96 | 33095.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 33570.00 | 33242.78 | 33160.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 33060.00 | 33256.18 | 33184.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 32955.00 | 33195.94 | 33163.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 33030.00 | 33195.94 | 33163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 33115.00 | 33150.04 | 33147.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 33045.00 | 33150.04 | 33147.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 32755.00 | 33071.03 | 33112.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 32470.00 | 32950.83 | 33053.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 32560.00 | 32410.80 | 32603.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 15:00:00 | 32560.00 | 32410.80 | 32603.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 32525.00 | 32433.64 | 32595.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 32655.00 | 32433.64 | 32595.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 32580.00 | 32462.91 | 32594.54 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 15:15:00 | 32760.00 | 32660.39 | 32655.03 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 32230.00 | 32574.32 | 32616.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 32120.00 | 32385.09 | 32510.94 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:45:00 | 29300.00 | 2025-05-13 09:15:00 | 29815.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-05-27 11:15:00 | 30270.00 | 2025-05-27 13:15:00 | 29990.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-30 15:15:00 | 29230.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-06-02 11:30:00 | 29270.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-03 09:15:00 | 29265.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-05 09:15:00 | 29735.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-05 10:00:00 | 29755.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-05 10:45:00 | 29765.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-06-09 11:00:00 | 29375.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-10 11:00:00 | 29450.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2025-06-10 11:45:00 | 29420.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.53% |
| SELL | retest2 | 2025-06-10 15:15:00 | 29355.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2025-06-11 11:00:00 | 29355.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2025-06-11 11:30:00 | 29370.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-06-11 12:15:00 | 29395.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest1 | 2025-07-07 09:15:00 | 29370.00 | 2025-07-14 09:15:00 | 29600.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-07-10 11:15:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-11 10:30:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-14 12:30:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-14 13:15:00 | 29705.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2025-07-15 09:15:00 | 30100.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2025-07-15 13:15:00 | 29765.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-07-16 09:15:00 | 29905.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-07-16 11:15:00 | 29840.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.84% |
| BUY | retest2 | 2025-07-18 14:15:00 | 29905.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-07-21 09:30:00 | 29965.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.40% |
| SELL | retest2 | 2025-08-13 10:45:00 | 30595.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-13 15:15:00 | 30450.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-08-18 11:45:00 | 30515.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest1 | 2025-08-25 09:15:00 | 30275.00 | 2025-08-25 13:15:00 | 30730.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest1 | 2025-08-25 10:30:00 | 30500.00 | 2025-08-25 13:15:00 | 30730.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-26 09:15:00 | 30265.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-08-26 10:30:00 | 30455.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-26 11:30:00 | 30400.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-26 15:00:00 | 30400.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-03 09:15:00 | 31260.00 | 2025-09-03 14:15:00 | 30990.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-10 10:00:00 | 30995.00 | 2025-09-10 13:15:00 | 30780.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-10 11:15:00 | 31005.00 | 2025-09-10 13:15:00 | 30780.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-17 12:00:00 | 30405.00 | 2025-09-25 14:15:00 | 28884.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 30400.00 | 2025-09-25 14:15:00 | 28880.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 14:45:00 | 30400.00 | 2025-09-25 14:15:00 | 28880.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:00:00 | 30405.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-09-18 10:00:00 | 30400.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-09-18 14:45:00 | 30400.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.85% |
| BUY | retest2 | 2025-10-16 12:30:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-16 13:15:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-16 14:00:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-17 09:30:00 | 29455.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-24 14:00:00 | 29725.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-11-10 09:45:00 | 36200.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2025-11-10 12:45:00 | 36025.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-11-10 14:00:00 | 36130.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest1 | 2025-11-11 13:15:00 | 36055.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-12 10:45:00 | 36150.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-12 14:00:00 | 36155.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-13 09:15:00 | 36350.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-13 10:00:00 | 36185.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-20 10:45:00 | 35690.00 | 2025-12-05 15:15:00 | 33905.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:15:00 | 35665.00 | 2025-12-05 15:15:00 | 33881.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 15:15:00 | 35640.00 | 2025-12-05 15:15:00 | 33858.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:45:00 | 35690.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-11-20 11:15:00 | 35665.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-11-20 15:15:00 | 35640.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest2 | 2025-12-16 12:00:00 | 35460.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-17 09:15:00 | 35565.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-17 11:00:00 | 35460.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-01-08 15:00:00 | 34750.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-09 13:00:00 | 34590.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-13 09:15:00 | 34590.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-23 11:30:00 | 33645.00 | 2026-01-28 14:15:00 | 33840.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-02-05 14:15:00 | 35775.00 | 2026-02-05 15:15:00 | 35180.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-02-09 09:15:00 | 35480.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2026-02-09 10:00:00 | 35425.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-02-09 11:00:00 | 35410.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2026-02-09 14:45:00 | 35520.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2026-02-17 11:15:00 | 35685.00 | 2026-02-18 11:15:00 | 36125.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-24 13:30:00 | 36015.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-24 14:30:00 | 35700.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-02-25 11:45:00 | 35865.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-03-17 13:45:00 | 33230.00 | 2026-03-23 09:15:00 | 31568.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 09:45:00 | 33250.00 | 2026-03-23 09:15:00 | 31587.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 13:45:00 | 33230.00 | 2026-03-24 12:15:00 | 31610.00 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2026-03-18 09:45:00 | 33250.00 | 2026-03-24 12:15:00 | 31610.00 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-04-01 12:30:00 | 30150.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-01 13:00:00 | 29955.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-08 09:30:00 | 30260.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-15 09:15:00 | 31630.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-04-17 09:15:00 | 31435.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2026-04-17 10:00:00 | 31490.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-04-28 13:30:00 | 33350.00 | 2026-04-29 11:15:00 | 32810.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-28 15:00:00 | 33420.00 | 2026-04-29 11:15:00 | 32810.00 | STOP_HIT | 1.00 | -1.83% |
