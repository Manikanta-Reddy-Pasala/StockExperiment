# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 7090.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 93 |
| ALERT1 | 53 |
| ALERT2 | 52 |
| ALERT2_SKIP | 27 |
| ALERT3 | 148 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 69 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 26 / 47
- **Target hits / Stop hits / Partials:** 0 / 72 / 1
- **Avg / median % per leg:** -0.40% / -0.71%
- **Sum % (uncompounded):** -29.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 18 | 40.0% | 0 | 45 | 0 | -0.29% | -13.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.69% | -6.8% |
| BUY @ 3rd Alert (retest2) | 41 | 18 | 43.9% | 0 | 41 | 0 | -0.15% | -6.3% |
| SELL (all) | 28 | 8 | 28.6% | 0 | 27 | 1 | -0.58% | -16.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 8 | 28.6% | 0 | 27 | 1 | -0.58% | -16.3% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.69% | -6.8% |
| retest2 (combined) | 69 | 26 | 37.7% | 0 | 68 | 1 | -0.33% | -22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 6875.00 | 6818.23 | 6812.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 15:15:00 | 6885.00 | 6831.58 | 6819.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 6834.00 | 6840.57 | 6827.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 6814.00 | 6835.26 | 6826.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 6807.50 | 6835.26 | 6826.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 6808.50 | 6829.91 | 6824.53 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 6781.00 | 6815.34 | 6818.55 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 6835.00 | 6820.12 | 6820.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 15:15:00 | 6801.00 | 6818.49 | 6819.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 6776.00 | 6809.99 | 6815.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 6805.00 | 6804.02 | 6811.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 6809.00 | 6805.02 | 6810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 6809.00 | 6805.02 | 6810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 6782.50 | 6800.51 | 6808.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 6760.00 | 6798.41 | 6806.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 6831.00 | 6804.93 | 6808.81 | SL hit (close>static) qty=1.00 sl=6812.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 6851.00 | 6814.14 | 6812.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 6915.50 | 6845.73 | 6830.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 13:15:00 | 6940.00 | 6941.71 | 6904.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:00:00 | 6940.00 | 6941.71 | 6904.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 6947.50 | 6944.74 | 6915.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 6980.00 | 6957.44 | 6926.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 6983.00 | 7079.94 | 7094.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 7104.00 | 7065.95 | 7082.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 7095.00 | 7071.76 | 7083.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 7114.00 | 7071.76 | 7083.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 7112.50 | 7079.91 | 7086.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 7123.00 | 7079.91 | 7086.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 7100.50 | 7084.03 | 7087.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 7078.50 | 7088.32 | 7089.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 7147.00 | 7112.70 | 7101.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 7089.50 | 7124.23 | 7113.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 7071.00 | 7113.58 | 7109.84 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 7061.50 | 7103.17 | 7105.45 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 7135.00 | 7107.54 | 7106.05 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 7099.50 | 7105.71 | 7106.18 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 7114.50 | 7107.47 | 7106.94 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 7091.00 | 7106.74 | 7106.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 7070.00 | 7099.39 | 7103.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 7140.50 | 7100.34 | 7098.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 7194.00 | 7119.07 | 7107.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 7227.00 | 7238.47 | 7205.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 7227.00 | 7238.47 | 7205.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 7205.00 | 7231.78 | 7205.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 7205.00 | 7231.78 | 7205.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 7211.50 | 7227.72 | 7205.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 7203.50 | 7227.72 | 7205.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 7332.50 | 7248.68 | 7217.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 7361.50 | 7270.44 | 7229.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 7364.00 | 7278.75 | 7237.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 7352.50 | 7291.30 | 7246.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 7380.00 | 7302.87 | 7259.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 7347.00 | 7388.80 | 7342.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 7359.50 | 7388.80 | 7342.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 7349.00 | 7380.84 | 7343.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 7345.50 | 7380.84 | 7343.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 7333.00 | 7371.27 | 7342.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 7333.00 | 7371.27 | 7342.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 7326.00 | 7362.22 | 7340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 7314.00 | 7347.37 | 7336.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 7204.00 | 7300.48 | 7315.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 7263.00 | 7128.75 | 7125.51 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 7106.00 | 7164.34 | 7164.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 7063.00 | 7144.07 | 7155.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 7028.00 | 7022.22 | 7060.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 6991.50 | 7022.22 | 7060.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 7014.00 | 7020.58 | 7056.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 6969.50 | 7009.85 | 7045.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 6977.00 | 6943.86 | 6957.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 7226.00 | 7046.71 | 7002.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 7443.00 | 7446.81 | 7391.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 7443.00 | 7446.81 | 7391.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 7378.00 | 7433.05 | 7389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 7374.00 | 7433.05 | 7389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 7409.00 | 7428.24 | 7391.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 7410.00 | 7428.24 | 7391.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 7411.50 | 7424.89 | 7393.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 7411.00 | 7417.94 | 7397.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 7443.50 | 7424.85 | 7402.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 7664.00 | 7690.85 | 7640.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 7710.00 | 7690.85 | 7640.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 7736.50 | 7702.57 | 7658.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 7553.00 | 7647.92 | 7647.90 | SL hit (close<static) qty=1.00 sl=7602.50 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 7517.50 | 7621.83 | 7636.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 7479.50 | 7547.96 | 7586.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 7530.00 | 7527.31 | 7565.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 7530.00 | 7527.31 | 7565.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7543.00 | 7520.18 | 7552.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 7410.50 | 7507.55 | 7543.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 7589.50 | 7526.46 | 7511.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 7473.00 | 7552.13 | 7534.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 7423.50 | 7526.40 | 7524.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 7552.50 | 7526.40 | 7524.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 12:15:00 | 7458.00 | 7494.65 | 7508.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 7423.50 | 7341.95 | 7387.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 7339.00 | 7341.36 | 7383.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:15:00 | 7171.00 | 7341.36 | 7383.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7270.39 | 7347.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 6914.00 | 7206.11 | 7310.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 15:15:00 | 6568.30 | 6646.27 | 6729.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 6585.00 | 6583.84 | 6668.76 | SL hit (close>ema200) qty=0.50 sl=6583.84 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 6716.50 | 6667.18 | 6666.77 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 6685.00 | 6696.72 | 6697.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 6638.50 | 6679.68 | 6689.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 6555.00 | 6533.06 | 6583.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 6555.00 | 6533.06 | 6583.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 6570.50 | 6540.55 | 6582.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 6571.50 | 6540.55 | 6582.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6584.50 | 6552.68 | 6577.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 6584.50 | 6552.68 | 6577.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 6571.50 | 6556.45 | 6576.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 6574.00 | 6556.45 | 6576.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 6528.50 | 6550.86 | 6572.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 6513.50 | 6550.86 | 6572.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 6634.50 | 6574.73 | 6578.41 | SL hit (close>static) qty=1.00 sl=6614.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 6673.00 | 6594.38 | 6587.01 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 6607.00 | 6610.56 | 6610.99 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 6634.50 | 6615.35 | 6613.12 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 6579.00 | 6619.51 | 6621.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 6505.00 | 6596.61 | 6611.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 6509.50 | 6494.70 | 6540.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 6509.50 | 6494.70 | 6540.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 6350.00 | 6344.79 | 6387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 6416.00 | 6344.79 | 6387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6468.00 | 6369.43 | 6394.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 6468.00 | 6369.43 | 6394.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6401.00 | 6375.74 | 6395.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 6448.00 | 6375.74 | 6395.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 6415.00 | 6388.36 | 6397.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 6420.00 | 6388.36 | 6397.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 6421.50 | 6394.37 | 6398.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 6421.50 | 6394.37 | 6398.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 6419.00 | 6399.29 | 6400.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 6395.00 | 6399.29 | 6400.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 6384.00 | 6393.63 | 6397.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 6372.50 | 6388.90 | 6395.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 6416.50 | 6388.71 | 6393.04 | SL hit (close>static) qty=1.00 sl=6402.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 6425.50 | 6398.99 | 6397.15 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 6385.00 | 6396.19 | 6396.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 6375.00 | 6389.76 | 6393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 6447.50 | 6389.03 | 6390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 6493.00 | 6409.82 | 6400.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 6538.50 | 6448.15 | 6420.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 11:15:00 | 6481.50 | 6488.41 | 6456.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:45:00 | 6477.50 | 6488.41 | 6456.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 6483.00 | 6487.33 | 6458.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 6506.50 | 6490.97 | 6462.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:30:00 | 6506.50 | 6493.77 | 6466.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 6345.00 | 6465.01 | 6458.43 | SL hit (close<static) qty=1.00 sl=6450.50 alert=retest2 |

### Cycle 30 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 6324.50 | 6436.91 | 6446.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 6260.00 | 6350.57 | 6397.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 6331.00 | 6323.03 | 6359.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:30:00 | 6326.50 | 6323.03 | 6359.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 6278.50 | 6288.22 | 6319.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 6278.50 | 6288.22 | 6319.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 6315.50 | 6299.63 | 6317.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 6315.00 | 6299.63 | 6317.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 6323.00 | 6304.30 | 6317.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 6319.00 | 6304.30 | 6317.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 6325.50 | 6308.54 | 6318.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 6325.50 | 6308.54 | 6318.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 6327.50 | 6312.33 | 6319.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 6330.50 | 6312.33 | 6319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 6285.50 | 6310.59 | 6317.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 6284.00 | 6310.59 | 6317.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 6304.50 | 6309.37 | 6316.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 6344.00 | 6324.48 | 6322.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 6377.00 | 6338.27 | 6329.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 6297.00 | 6364.86 | 6369.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 6272.00 | 6346.29 | 6360.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 6310.00 | 6300.74 | 6325.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 6310.00 | 6300.74 | 6325.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 6326.50 | 6305.90 | 6325.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 6326.50 | 6305.90 | 6325.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6323.00 | 6309.32 | 6325.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6323.50 | 6309.32 | 6325.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 6333.00 | 6314.05 | 6326.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:15:00 | 6331.00 | 6314.05 | 6326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 6331.00 | 6317.44 | 6326.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 6296.50 | 6317.44 | 6326.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 6335.00 | 6323.76 | 6328.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:15:00 | 6337.50 | 6323.76 | 6328.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 6342.50 | 6327.51 | 6329.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 6337.00 | 6327.51 | 6329.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 6326.50 | 6328.83 | 6329.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:15:00 | 6335.50 | 6328.83 | 6329.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 6353.50 | 6333.76 | 6331.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 6360.00 | 6339.01 | 6334.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 6410.00 | 6417.22 | 6398.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 6391.50 | 6417.22 | 6398.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 6377.50 | 6409.27 | 6396.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 6374.50 | 6409.27 | 6396.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 6326.00 | 6392.62 | 6389.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 6326.00 | 6392.62 | 6389.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 6361.00 | 6386.30 | 6387.19 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 6394.00 | 6387.84 | 6387.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 6468.00 | 6405.30 | 6395.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 12:15:00 | 6420.00 | 6431.35 | 6412.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 6420.00 | 6431.35 | 6412.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 6435.00 | 6436.60 | 6421.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 6463.00 | 6436.60 | 6421.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 6455.00 | 6439.88 | 6424.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 6448.50 | 6445.72 | 6432.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 6423.50 | 6476.55 | 6488.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 6283.50 | 6269.49 | 6306.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 6283.50 | 6269.49 | 6306.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 6049.00 | 6019.97 | 6052.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 6049.00 | 6019.97 | 6052.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 6050.00 | 6025.97 | 6052.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 6015.50 | 6025.97 | 6052.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 6100.00 | 6045.12 | 6051.60 | SL hit (close>static) qty=1.00 sl=6080.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 6194.00 | 6074.89 | 6064.54 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 6050.00 | 6081.19 | 6083.69 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 6100.50 | 6081.72 | 6080.88 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 6049.50 | 6077.60 | 6079.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 6034.50 | 6068.98 | 6075.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 6061.00 | 6008.81 | 6024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 6000.50 | 6007.15 | 6022.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 5995.00 | 6007.15 | 6022.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 5992.50 | 6013.63 | 6020.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 5991.00 | 6010.90 | 6018.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 5987.00 | 6006.82 | 6015.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 5995.00 | 6000.71 | 6009.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 5987.50 | 6000.71 | 6009.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 5943.00 | 5989.16 | 6003.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 5917.00 | 5974.53 | 5995.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 5859.00 | 5842.01 | 5876.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 5853.50 | 5881.32 | 5882.83 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5913.00 | 5887.65 | 5885.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 5922.50 | 5898.84 | 5891.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 5917.00 | 5922.95 | 5908.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 11:45:00 | 5916.00 | 5922.95 | 5908.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5905.00 | 5917.77 | 5908.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 5905.00 | 5917.77 | 5908.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 5940.00 | 5922.21 | 5911.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 5967.00 | 5925.92 | 5915.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 5925.00 | 5904.02 | 5903.13 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5887.00 | 5904.19 | 5904.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 5835.50 | 5880.91 | 5891.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 5831.50 | 5799.37 | 5819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 5865.00 | 5812.49 | 5823.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 5868.00 | 5812.49 | 5823.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 5894.00 | 5828.79 | 5829.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 5894.00 | 5828.79 | 5829.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 5911.00 | 5845.24 | 5837.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 5940.00 | 5872.95 | 5851.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 5829.50 | 5878.17 | 5867.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 5825.00 | 5867.54 | 5863.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 5820.00 | 5867.54 | 5863.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 5802.00 | 5854.43 | 5858.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 5764.00 | 5825.72 | 5843.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 5720.00 | 5715.48 | 5756.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:30:00 | 5725.50 | 5715.48 | 5756.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5758.50 | 5720.78 | 5740.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 5788.00 | 5720.78 | 5740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 5777.50 | 5732.13 | 5743.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 5750.00 | 5745.65 | 5747.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 5744.50 | 5748.72 | 5749.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 5827.00 | 5769.72 | 5758.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 6100.00 | 6120.48 | 6051.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 6061.00 | 6120.48 | 6051.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 6080.00 | 6121.61 | 6087.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 6087.00 | 6121.61 | 6087.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 6069.50 | 6111.19 | 6086.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 6069.50 | 6111.19 | 6086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 6094.00 | 6107.75 | 6086.91 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 5976.50 | 6061.35 | 6071.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 5976.00 | 6044.28 | 6062.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 5943.50 | 5934.31 | 5968.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 5970.00 | 5941.60 | 5961.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 5980.00 | 5949.28 | 5962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 5979.00 | 5949.28 | 5962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 5927.00 | 5945.42 | 5958.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5893.50 | 5945.82 | 5954.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 5882.50 | 5926.27 | 5936.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 5759.00 | 5779.26 | 5780.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 5710.00 | 5765.41 | 5774.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 5740.00 | 5732.19 | 5749.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 5740.00 | 5732.19 | 5749.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5745.00 | 5734.75 | 5749.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 5752.00 | 5734.75 | 5749.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5855.00 | 5758.80 | 5759.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 5855.00 | 5758.80 | 5759.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5858.00 | 5778.64 | 5768.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 5895.50 | 5858.52 | 5839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 5868.00 | 5890.18 | 5865.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 5845.00 | 5881.14 | 5863.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 5840.50 | 5881.14 | 5863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 5851.00 | 5875.11 | 5862.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 5855.00 | 5861.23 | 5858.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 5785.00 | 5835.95 | 5847.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 5878.00 | 5764.11 | 5785.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 5850.00 | 5781.29 | 5791.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 5764.00 | 5781.93 | 5790.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 6104.00 | 5860.40 | 5824.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 5965.00 | 5990.31 | 5947.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 5967.00 | 5990.31 | 5947.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 6040.00 | 6000.24 | 5956.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 5975.00 | 6000.24 | 5956.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 5963.50 | 5997.66 | 5962.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 5961.00 | 5997.66 | 5962.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 5924.00 | 5982.93 | 5959.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 5924.00 | 5982.93 | 5959.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 5923.50 | 5971.04 | 5956.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 5918.00 | 5971.04 | 5956.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 6047.50 | 6043.62 | 6014.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:30:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 6078.00 | 6051.99 | 6023.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 6065.00 | 6111.33 | 6093.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 6039.00 | 6096.65 | 6089.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 6039.00 | 6096.65 | 6089.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6052.00 | 6087.72 | 6086.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 6084.50 | 6087.72 | 6086.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 6111.00 | 6089.78 | 6087.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 6130.50 | 6097.92 | 6091.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 6096.00 | 6123.13 | 6108.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 6094.00 | 6117.30 | 6107.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 6094.00 | 6117.30 | 6107.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 6111.50 | 6115.77 | 6108.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 6141.00 | 6123.82 | 6112.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 6125.50 | 6139.55 | 6125.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:00:00 | 6126.00 | 6130.44 | 6124.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 6155.50 | 6121.30 | 6120.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 6127.50 | 6153.00 | 6142.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 6115.00 | 6153.00 | 6142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | SL hit (close<static) qty=1.00 sl=6094.50 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 6092.00 | 6130.40 | 6133.51 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 6154.00 | 6132.97 | 6132.27 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 6077.00 | 6126.22 | 6130.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 6072.00 | 6117.50 | 6126.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 6179.00 | 6116.60 | 6122.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 6186.50 | 6130.58 | 6128.13 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 6079.00 | 6123.29 | 6125.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 6009.50 | 6062.61 | 6087.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 6078.00 | 6053.51 | 6077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 6184.50 | 6079.71 | 6087.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 6184.50 | 6079.71 | 6087.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 6197.50 | 6103.27 | 6097.64 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 6056.00 | 6115.10 | 6116.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 6000.00 | 6058.04 | 6086.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 15:15:00 | 6015.50 | 6010.46 | 6044.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 5955.00 | 6010.46 | 6044.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 5801.00 | 5731.38 | 5795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 5801.00 | 5731.38 | 5795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 5829.50 | 5751.01 | 5798.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 5841.00 | 5751.01 | 5798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5869.50 | 5774.70 | 5805.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:30:00 | 5775.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 5771.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 5763.00 | 5787.33 | 5802.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 5755.00 | 5792.28 | 5801.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 5809.00 | 5795.62 | 5801.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 5809.00 | 5795.62 | 5801.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 5780.00 | 5792.50 | 5799.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:15:00 | 5865.50 | 5792.50 | 5799.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5814.00 | 5796.80 | 5801.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 5872.50 | 5821.49 | 5812.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:30:00 | 5809.50 | 5828.80 | 5819.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 5836.00 | 5830.24 | 5820.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 5845.50 | 5830.24 | 5820.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5851.50 | 5836.93 | 5825.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 5886.00 | 6031.06 | 6073.86 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6403.50 | 6058.92 | 6057.39 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 6555.00 | 6581.38 | 6582.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 15:15:00 | 6539.50 | 6567.49 | 6575.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 6599.50 | 6570.09 | 6576.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 6594.50 | 6574.97 | 6577.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 6597.00 | 6574.97 | 6577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 6686.50 | 6597.28 | 6587.78 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 6552.00 | 6605.10 | 6611.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 15:15:00 | 6505.00 | 6564.85 | 6582.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 6514.00 | 6498.87 | 6530.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 6514.00 | 6498.87 | 6530.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 6515.00 | 6502.09 | 6529.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 6524.00 | 6502.09 | 6529.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 6509.50 | 6504.76 | 6523.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 6520.00 | 6504.76 | 6523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 6546.50 | 6513.11 | 6525.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 6546.50 | 6513.11 | 6525.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 6544.00 | 6519.29 | 6527.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 6559.50 | 6519.29 | 6527.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 6574.00 | 6530.23 | 6531.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 6582.00 | 6530.23 | 6531.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 6561.00 | 6536.38 | 6534.45 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 6522.00 | 6531.41 | 6532.58 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 6553.50 | 6533.36 | 6533.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 6587.50 | 6544.19 | 6538.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 6572.50 | 6562.14 | 6549.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 6699.00 | 6589.51 | 6563.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:30:00 | 6567.50 | 6589.51 | 6563.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6575.00 | 6593.41 | 6570.06 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 6526.00 | 6554.58 | 6557.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 6383.50 | 6520.36 | 6541.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6578.00 | 6500.21 | 6490.58 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 6467.50 | 6483.61 | 6484.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 6423.00 | 6469.55 | 6477.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 10:15:00 | 6124.00 | 6107.83 | 6196.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 6124.00 | 6107.83 | 6196.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 6179.00 | 6135.89 | 6188.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 6180.00 | 6135.89 | 6188.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 6207.50 | 6150.21 | 6190.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 6185.00 | 6150.21 | 6190.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 6200.00 | 6160.17 | 6191.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 6101.00 | 6160.17 | 6191.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 6290.00 | 6178.17 | 6190.79 | SL hit (close>static) qty=1.00 sl=6227.50 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 6303.00 | 6203.13 | 6200.99 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 6108.00 | 6205.26 | 6208.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 6085.00 | 6181.21 | 6197.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 6428.00 | 6188.20 | 6192.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 6448.00 | 6240.16 | 6215.64 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 6173.50 | 6218.21 | 6224.31 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 6342.50 | 6247.93 | 6236.54 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 6228.00 | 6261.53 | 6262.32 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 6291.50 | 6267.52 | 6264.98 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6186.50 | 6253.91 | 6262.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 6068.00 | 6189.25 | 6228.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6143.00 | 6134.30 | 6177.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 6150.00 | 6134.30 | 6177.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6155.00 | 6138.44 | 6175.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 6196.00 | 6138.44 | 6175.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6182.50 | 6147.25 | 6176.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 6182.50 | 6147.25 | 6176.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 6131.50 | 6144.10 | 6172.26 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 6245.00 | 6190.02 | 6186.28 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 6175.00 | 6198.91 | 6200.41 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 6286.50 | 6218.04 | 6208.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 6370.50 | 6248.53 | 6223.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 6366.00 | 6394.50 | 6330.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 6316.00 | 6394.50 | 6330.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 6318.50 | 6379.30 | 6329.64 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 6160.50 | 6292.18 | 6305.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 6139.00 | 6261.55 | 6290.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 6374.00 | 6265.61 | 6252.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 6400.00 | 6340.31 | 6298.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 6377.00 | 6398.46 | 6353.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 6501.00 | 6398.46 | 6353.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 6479.50 | 6411.57 | 6363.25 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 12:30:00 | 6485.00 | 6438.90 | 6389.09 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 14:00:00 | 6500.00 | 6451.12 | 6399.17 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | SL hit (close<ema400) qty=1.00 sl=6407.28 alert=retest1 |

### Cycle 90 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 6333.00 | 6392.65 | 6393.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 6299.00 | 6363.50 | 6379.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 6429.00 | 6376.60 | 6383.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 6486.00 | 6398.48 | 6393.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 6565.00 | 6444.43 | 6415.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 09:15:00 | 6566.50 | 6573.45 | 6524.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 6566.50 | 6573.45 | 6524.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 6575.00 | 6581.28 | 6552.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 6550.50 | 6581.28 | 6552.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 6543.50 | 6574.96 | 6558.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:30:00 | 6540.00 | 6574.96 | 6558.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 6549.00 | 6569.77 | 6558.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 6549.00 | 6569.77 | 6558.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 6532.00 | 6562.21 | 6555.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 6562.00 | 6562.21 | 6555.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 6567.00 | 6558.30 | 6554.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 12:30:00 | 6567.00 | 6568.61 | 6560.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 6596.50 | 6568.50 | 6562.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 6591.50 | 6573.10 | 6565.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 6659.50 | 6610.55 | 6587.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 6650.00 | 6624.75 | 6598.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:15:00 | 6668.50 | 6700.84 | 6692.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 6716.50 | 6694.64 | 6690.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 6700.00 | 6703.34 | 6696.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 6700.00 | 6703.34 | 6696.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 6813.00 | 6724.73 | 6707.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 6879.00 | 6724.73 | 6707.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 6849.50 | 6857.39 | 6801.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 6830.00 | 6820.47 | 6800.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 6825.00 | 6788.10 | 6786.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 6964.50 | 6823.38 | 6802.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 6888.00 | 6934.57 | 6888.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 6893.00 | 6920.96 | 6890.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 6893.00 | 6920.96 | 6890.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 6895.50 | 6915.87 | 6890.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 6895.50 | 6915.87 | 6890.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 7005.00 | 7053.17 | 7016.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 7093.50 | 7053.17 | 7016.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:15:00 | 6786.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-13 09:15:00 | 6771.50 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-19 09:15:00 | 6760.00 | 2025-05-19 09:15:00 | 6831.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-22 11:45:00 | 6980.00 | 2025-05-27 12:15:00 | 7074.50 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-05-28 13:30:00 | 7078.50 | 2025-05-28 14:15:00 | 7137.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-06 14:30:00 | 7361.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-06-09 09:15:00 | 7364.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-09 10:15:00 | 7352.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-09 12:15:00 | 7380.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-06-20 11:30:00 | 6969.50 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-06-24 13:15:00 | 6977.00 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-07-01 13:15:00 | 7410.00 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-07-01 14:00:00 | 7411.50 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2025-07-02 10:00:00 | 7411.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-07-02 10:45:00 | 7443.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-08 09:15:00 | 7710.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-07-08 11:45:00 | 7736.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-11 11:15:00 | 7410.50 | 2025-07-14 15:15:00 | 7575.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-16 09:15:00 | 7552.50 | 2025-07-16 10:15:00 | 7473.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-24 15:15:00 | 6568.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-25 12:15:00 | 6585.00 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-08-05 10:15:00 | 6513.50 | 2025-08-05 12:15:00 | 6634.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-20 11:30:00 | 6372.50 | 2025-08-20 14:15:00 | 6416.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-25 13:45:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-08-25 14:30:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-16 10:15:00 | 6463.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-16 10:45:00 | 6455.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 15:00:00 | 6448.50 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-01 09:15:00 | 6015.50 | 2025-10-01 13:15:00 | 6100.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-10 11:15:00 | 5995.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-10-13 09:45:00 | 5992.50 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-10-13 11:15:00 | 5991.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-10-13 11:45:00 | 5987.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-14 10:30:00 | 5917.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-16 10:15:00 | 5859.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-24 10:15:00 | 5967.00 | 2025-10-24 13:15:00 | 5874.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-11 13:45:00 | 5750.00 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-11 15:15:00 | 5744.50 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-27 09:15:00 | 5893.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-11-28 09:15:00 | 5882.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-12-15 15:15:00 | 5855.00 | 2025-12-16 09:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-18 09:30:00 | 5764.00 | 2025-12-18 15:15:00 | 5835.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 09:30:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 10:15:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 11:00:00 | 6078.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-30 12:30:00 | 6065.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-12-31 09:15:00 | 6084.50 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-01 13:30:00 | 6141.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-02 10:30:00 | 6125.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-02 14:00:00 | 6126.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-05 09:15:00 | 6155.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-22 13:30:00 | 5775.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-22 14:00:00 | 5771.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-22 15:00:00 | 5763.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-23 12:00:00 | 5755.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-27 15:15:00 | 5845.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2026-01-28 10:00:00 | 5851.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2026-03-12 09:15:00 | 6101.00 | 2026-03-12 11:15:00 | 6290.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest1 | 2026-04-10 09:15:00 | 6501.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2026-04-10 10:15:00 | 6479.50 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2026-04-10 12:30:00 | 6485.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest1 | 2026-04-10 14:00:00 | 6500.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-21 09:15:00 | 6562.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2026-04-21 10:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-21 12:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-22 09:15:00 | 6596.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-04-22 15:00:00 | 6659.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2026-04-23 10:00:00 | 6650.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2026-04-27 10:15:00 | 6668.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-04-27 12:00:00 | 6716.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-04-28 10:15:00 | 6879.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-04-29 09:30:00 | 6849.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-29 15:00:00 | 6830.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.63% |
