# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 7765.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 50 |
| ALERT2 | 48 |
| ALERT2_SKIP | 28 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 91 |
| PARTIAL | 18 |
| TARGET_HIT | 2 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 59
- **Target hits / Stop hits / Partials:** 2 / 89 / 18
- **Avg / median % per leg:** 0.99% / -0.34%
- **Sum % (uncompounded):** 107.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 6 | 14.0% | 2 | 41 | 0 | -0.22% | -9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 6 | 14.0% | 2 | 41 | 0 | -0.22% | -9.4% |
| SELL (all) | 66 | 44 | 66.7% | 0 | 48 | 18 | 1.78% | 117.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 66 | 44 | 66.7% | 0 | 48 | 18 | 1.78% | 117.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 109 | 50 | 45.9% | 2 | 89 | 18 | 0.99% | 107.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 6381.00 | 6169.64 | 6155.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 6475.00 | 6377.04 | 6304.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 7235.00 | 7247.10 | 7118.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 7235.00 | 7247.10 | 7118.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 7224.00 | 7249.41 | 7162.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:30:00 | 7228.50 | 7243.53 | 7167.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:30:00 | 7225.50 | 7231.12 | 7168.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 7106.00 | 7206.10 | 7163.26 | SL hit (close<static) qty=1.00 sl=7150.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 7106.00 | 7206.10 | 7163.26 | SL hit (close<static) qty=1.00 sl=7150.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 7112.50 | 7138.65 | 7140.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 7051.00 | 7121.12 | 7132.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 7085.00 | 7042.59 | 7076.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 7085.00 | 7042.59 | 7076.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 7026.00 | 7039.27 | 7071.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 7080.00 | 7039.27 | 7071.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 7053.00 | 7042.02 | 7069.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 7055.50 | 7042.02 | 7069.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 7040.00 | 7043.43 | 7063.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 7040.00 | 7043.43 | 7063.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 7035.00 | 7041.75 | 7061.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 7371.00 | 7041.75 | 7061.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 7512.00 | 7135.80 | 7102.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 7679.00 | 7244.44 | 7154.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 14:15:00 | 7505.00 | 7511.10 | 7404.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 7505.00 | 7511.10 | 7404.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 7521.00 | 7553.94 | 7525.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 7512.50 | 7553.94 | 7525.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 7535.50 | 7550.25 | 7526.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 7616.50 | 7556.70 | 7531.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 7480.00 | 7536.14 | 7527.97 | SL hit (close<static) qty=1.00 sl=7522.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 7500.00 | 7521.27 | 7522.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 7473.00 | 7511.62 | 7518.06 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 7596.50 | 7528.59 | 7525.19 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 7474.00 | 7524.36 | 7528.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 7450.00 | 7509.48 | 7521.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 7565.00 | 7520.59 | 7525.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 7565.00 | 7520.59 | 7525.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 7515.00 | 7519.47 | 7524.74 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 7555.00 | 7530.27 | 7528.21 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 7490.00 | 7522.99 | 7526.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 7470.50 | 7504.80 | 7516.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 7500.00 | 7479.55 | 7496.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 7478.50 | 7479.55 | 7496.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 7472.50 | 7478.65 | 7492.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 7480.00 | 7478.92 | 7491.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:45:00 | 7472.50 | 7477.24 | 7489.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 7478.50 | 7477.49 | 7488.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 7478.50 | 7477.49 | 7488.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 7480.00 | 7477.99 | 7488.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 7484.50 | 7477.99 | 7488.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 7456.00 | 7473.59 | 7485.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 10:45:00 | 7420.50 | 7461.48 | 7478.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:45:00 | 7435.00 | 7393.67 | 7418.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 7425.00 | 7393.67 | 7418.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7104.57 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7098.88 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7106.00 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 7098.88 | 7239.01 | 7307.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7049.47 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7063.25 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 7053.75 | 7153.27 | 7227.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 7140.00 | 7110.73 | 7173.03 | SL hit (close>ema200) qty=0.50 sl=7110.73 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 6694.00 | 6646.16 | 6644.42 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 6574.50 | 6638.26 | 6645.43 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 6670.50 | 6624.05 | 6622.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 6686.00 | 6644.43 | 6632.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 6620.50 | 6641.58 | 6633.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 6620.50 | 6641.58 | 6633.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 6620.00 | 6637.26 | 6632.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 6625.00 | 6631.41 | 6629.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 6595.50 | 6624.23 | 6626.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 6595.50 | 6624.23 | 6626.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 6585.50 | 6607.28 | 6617.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 6604.50 | 6602.07 | 6612.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 6609.00 | 6602.07 | 6612.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 6616.50 | 6604.95 | 6612.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 6626.00 | 6604.95 | 6612.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 6607.00 | 6605.36 | 6612.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 6712.00 | 6605.36 | 6612.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 6638.00 | 6611.89 | 6614.41 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 6670.00 | 6623.51 | 6619.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 6757.00 | 6675.80 | 6649.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 6802.50 | 6803.12 | 6747.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:45:00 | 6812.50 | 6803.12 | 6747.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 6741.00 | 6783.55 | 6758.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 6741.00 | 6783.55 | 6758.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 6737.50 | 6774.34 | 6756.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 6740.00 | 6774.34 | 6756.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 6741.00 | 6767.67 | 6755.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 6723.50 | 6767.67 | 6755.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 6772.00 | 6775.00 | 6762.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 6772.00 | 6775.00 | 6762.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 6778.50 | 6775.70 | 6763.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 6808.00 | 6775.70 | 6763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 6795.00 | 6779.56 | 6766.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 6844.00 | 6815.33 | 6797.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 6869.00 | 6816.61 | 6801.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:00:00 | 6844.00 | 6822.55 | 6806.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 6840.00 | 6828.03 | 6812.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 6840.50 | 6829.16 | 6815.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 6811.00 | 6829.16 | 6815.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 6812.50 | 6829.00 | 6817.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 6824.00 | 6829.00 | 6817.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 6800.00 | 6823.20 | 6816.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 6802.00 | 6823.20 | 6816.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 6804.50 | 6819.46 | 6815.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 6790.50 | 6819.46 | 6815.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 6771.00 | 6806.42 | 6809.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 6756.00 | 6796.33 | 6804.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 6704.50 | 6704.22 | 6735.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:30:00 | 6720.00 | 6704.22 | 6735.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 6688.50 | 6686.60 | 6713.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 6690.00 | 6686.60 | 6713.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 6716.50 | 6692.58 | 6713.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 6699.00 | 6692.58 | 6713.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 6680.50 | 6690.16 | 6710.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 6675.00 | 6690.16 | 6710.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:00:00 | 6675.00 | 6611.46 | 6632.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 6570.00 | 6536.65 | 6535.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 6570.00 | 6536.65 | 6535.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 6570.00 | 6536.65 | 6535.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6610.50 | 6551.42 | 6542.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6546.50 | 6589.80 | 6572.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 6615.00 | 6589.03 | 6575.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 6587.00 | 6582.43 | 6578.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 6561.00 | 6574.24 | 6574.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 6561.00 | 6574.24 | 6574.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 6561.00 | 6574.24 | 6574.85 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 6588.00 | 6576.99 | 6576.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 15:15:00 | 6600.50 | 6585.49 | 6580.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 6548.50 | 6578.09 | 6577.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 6548.50 | 6578.09 | 6577.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 6536.00 | 6569.67 | 6573.80 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 6609.50 | 6577.64 | 6577.04 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 6535.00 | 6579.86 | 6580.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6498.00 | 6558.63 | 6570.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 6547.00 | 6454.25 | 6483.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 6547.00 | 6454.25 | 6483.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 6544.00 | 6472.20 | 6489.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 6401.00 | 6472.20 | 6489.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 6276.50 | 6241.59 | 6241.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 6276.50 | 6241.59 | 6241.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 6392.00 | 6271.67 | 6254.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 14:15:00 | 6400.00 | 6405.19 | 6346.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 6400.00 | 6405.19 | 6346.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 6370.00 | 6395.24 | 6370.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 6370.00 | 6395.24 | 6370.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 6353.00 | 6386.79 | 6368.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 6389.00 | 6383.64 | 6368.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 6399.00 | 6375.57 | 6370.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 6326.00 | 6365.66 | 6366.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 6326.00 | 6365.66 | 6366.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 6326.00 | 6365.66 | 6366.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 6321.00 | 6345.72 | 6356.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 6329.00 | 6326.33 | 6341.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 6288.50 | 6317.55 | 6334.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 6279.50 | 6317.55 | 6334.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 6295.50 | 6307.83 | 6326.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 6222.00 | 6307.57 | 6324.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | SL hit (close>static) qty=1.00 sl=6343.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | SL hit (close>static) qty=1.00 sl=6343.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | SL hit (close>static) qty=1.00 sl=6343.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 6400.00 | 6302.93 | 6316.33 | SL hit (close>static) qty=1.00 sl=6343.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 6400.00 | 6302.93 | 6316.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 6559.00 | 6354.14 | 6338.39 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 6355.00 | 6366.57 | 6368.04 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 6401.00 | 6370.88 | 6369.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 6406.00 | 6383.42 | 6376.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 6347.00 | 6376.14 | 6373.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 6341.50 | 6376.14 | 6373.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 6358.50 | 6372.61 | 6372.23 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 6360.00 | 6370.92 | 6371.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 6387.00 | 6374.14 | 6372.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 6405.00 | 6380.31 | 6375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 6376.50 | 6387.77 | 6381.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 6367.00 | 6387.77 | 6381.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 6408.00 | 6391.82 | 6384.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 6429.00 | 6398.44 | 6390.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 6460.50 | 6412.60 | 6398.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 6430.00 | 6434.48 | 6423.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 6435.00 | 6429.15 | 6424.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 6415.50 | 6426.42 | 6423.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 6415.50 | 6426.42 | 6423.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 6392.00 | 6419.53 | 6420.67 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 6456.00 | 6422.47 | 6421.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 6487.00 | 6435.37 | 6427.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 6470.50 | 6474.87 | 6453.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:45:00 | 6468.00 | 6474.87 | 6453.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 6460.00 | 6471.89 | 6454.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 6453.50 | 6471.89 | 6454.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 6456.00 | 6468.71 | 6454.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 6445.50 | 6468.71 | 6454.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 6445.50 | 6464.07 | 6453.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 6489.00 | 6465.96 | 6455.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 6474.00 | 6465.41 | 6456.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 6474.00 | 6466.34 | 6458.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 6474.00 | 6466.34 | 6458.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 6446.00 | 6464.22 | 6459.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 6511.00 | 6464.22 | 6459.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 6442.50 | 6473.30 | 6466.11 | SL hit (close<static) qty=1.00 sl=6446.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 6437.00 | 6466.04 | 6463.46 | SL hit (close<static) qty=1.00 sl=6438.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 6437.00 | 6466.04 | 6463.46 | SL hit (close<static) qty=1.00 sl=6438.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 6437.00 | 6466.04 | 6463.46 | SL hit (close<static) qty=1.00 sl=6438.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 6437.00 | 6466.04 | 6463.46 | SL hit (close<static) qty=1.00 sl=6438.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 6414.00 | 6455.63 | 6458.96 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 6493.00 | 6460.06 | 6459.45 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 6439.50 | 6457.62 | 6459.09 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 6528.00 | 6464.52 | 6460.50 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 6440.00 | 6489.61 | 6493.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 6434.00 | 6468.01 | 6479.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 6424.50 | 6392.68 | 6423.67 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 6474.50 | 6417.94 | 6412.68 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 6393.00 | 6412.99 | 6414.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 6376.50 | 6405.69 | 6411.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 6381.00 | 6380.52 | 6392.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 6381.00 | 6380.52 | 6392.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 6352.50 | 6371.87 | 6386.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 6315.00 | 6362.50 | 6380.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:00:00 | 6306.50 | 6330.21 | 6357.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 6285.00 | 6330.67 | 6355.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 6299.00 | 6324.33 | 6350.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 6314.00 | 6258.80 | 6271.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 6314.00 | 6258.80 | 6271.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 6330.00 | 6273.04 | 6276.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 6296.00 | 6273.04 | 6276.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 6204.00 | 6158.17 | 6155.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 6225.00 | 6190.82 | 6174.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 6195.00 | 6195.92 | 6180.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 6195.00 | 6195.92 | 6180.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 6189.00 | 6194.54 | 6181.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 6191.00 | 6194.54 | 6181.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 6173.00 | 6190.23 | 6180.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 6220.00 | 6181.15 | 6179.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 6230.00 | 6184.92 | 6181.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 6206.50 | 6188.17 | 6184.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 6229.00 | 6194.23 | 6188.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6188.00 | 6192.98 | 6188.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 6188.00 | 6192.98 | 6188.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6188.00 | 6191.99 | 6188.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 6210.00 | 6187.88 | 6186.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 6197.00 | 6210.82 | 6203.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 6158.50 | 6194.94 | 6197.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 6150.00 | 6180.36 | 6190.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 6144.50 | 6120.90 | 6143.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 6130.50 | 6120.90 | 6143.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 6135.00 | 6123.72 | 6142.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 6105.50 | 6123.72 | 6142.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6103.00 | 6119.58 | 6138.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 6089.00 | 6111.02 | 6125.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:30:00 | 6080.00 | 6100.78 | 6115.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 6082.50 | 6096.92 | 6110.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 6082.00 | 6095.63 | 6108.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 6095.00 | 6091.40 | 6103.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 6117.00 | 6091.40 | 6103.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 6099.00 | 6088.50 | 6097.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 6078.00 | 6094.10 | 6099.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 6086.00 | 6092.48 | 6097.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 6085.00 | 6085.58 | 6094.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 6089.00 | 6075.96 | 6083.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 6080.00 | 6076.77 | 6082.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 6070.00 | 6076.77 | 6082.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5784.55 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5776.00 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5778.38 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5777.90 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5774.10 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5781.70 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5780.75 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5784.55 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 5766.50 | 5834.98 | 5873.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 5795.50 | 5794.98 | 5823.89 | SL hit (close>ema200) qty=0.50 sl=5794.98 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 5870.50 | 5747.84 | 5746.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 5970.00 | 5792.28 | 5767.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5972.00 | 6082.06 | 6001.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 5972.00 | 6082.06 | 6001.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 5953.00 | 6056.25 | 5997.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 5953.00 | 6056.25 | 5997.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 5978.00 | 6030.16 | 5995.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 5974.50 | 6030.16 | 5995.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5940.00 | 5996.29 | 5986.90 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 5875.00 | 5972.03 | 5976.73 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 5952.00 | 5928.33 | 5926.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 5992.50 | 5941.17 | 5932.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5983.50 | 6008.43 | 5985.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5983.50 | 6008.43 | 5985.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5972.50 | 6001.24 | 5984.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 5972.50 | 6001.24 | 5984.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 5927.00 | 5986.39 | 5979.40 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 5934.00 | 5970.97 | 5973.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 5890.00 | 5951.98 | 5964.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 5780.00 | 5777.74 | 5826.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 5780.00 | 5777.74 | 5826.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 5846.00 | 5794.61 | 5822.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 5846.00 | 5794.61 | 5822.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 5805.00 | 5796.68 | 5820.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 5799.00 | 5802.08 | 5818.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 5781.50 | 5804.56 | 5818.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 5787.00 | 5804.56 | 5818.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 5941.00 | 5825.75 | 5824.14 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 5812.00 | 5850.62 | 5852.78 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 5864.00 | 5855.77 | 5854.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 5885.00 | 5861.62 | 5857.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 5896.50 | 5915.47 | 5894.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 5896.50 | 5915.47 | 5894.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 5881.00 | 5908.58 | 5893.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 5870.50 | 5908.58 | 5893.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 5840.00 | 5894.86 | 5888.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 5916.00 | 5894.86 | 5888.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 5899.50 | 5895.79 | 5889.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 5952.00 | 6036.77 | 6038.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 5952.00 | 6036.77 | 6038.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 5952.00 | 6036.77 | 6038.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 5925.00 | 5982.27 | 6006.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 5929.00 | 5926.75 | 5954.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 5921.50 | 5926.75 | 5954.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 5923.00 | 5918.26 | 5943.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 5900.50 | 5917.61 | 5940.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:15:00 | 5910.00 | 5911.92 | 5931.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 5910.00 | 5912.44 | 5930.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 5910.50 | 5916.43 | 5927.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 5880.00 | 5892.68 | 5909.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 5859.00 | 5884.55 | 5904.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 5855.00 | 5879.04 | 5900.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 5848.00 | 5876.53 | 5897.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5953.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5953.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5953.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5953.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5947.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5947.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 5990.00 | 5887.95 | 5888.30 | SL hit (close>static) qty=1.00 sl=5947.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 5984.50 | 5907.26 | 5897.04 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 5902.50 | 5918.72 | 5918.99 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 5950.00 | 5920.47 | 5919.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 5959.50 | 5929.40 | 5923.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 5925.50 | 5932.07 | 5926.70 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 5907.50 | 5923.31 | 5923.41 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 5937.00 | 5921.86 | 5921.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5955.00 | 5928.49 | 5924.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 5913.00 | 5932.03 | 5927.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 5913.00 | 5932.03 | 5927.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 5911.50 | 5927.93 | 5926.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 5912.00 | 5927.93 | 5926.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5921.00 | 5926.54 | 5925.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 5921.00 | 5926.54 | 5925.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 5910.00 | 5923.23 | 5924.45 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 5956.50 | 5923.52 | 5923.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 15:15:00 | 5970.00 | 5941.13 | 5932.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 5954.00 | 5964.66 | 5952.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 5951.50 | 5964.66 | 5952.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 5931.00 | 5957.93 | 5950.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 5931.00 | 5957.93 | 5950.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 5910.00 | 5948.34 | 5946.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 5910.00 | 5948.34 | 5946.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 5895.00 | 5937.67 | 5942.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 5886.50 | 5924.93 | 5935.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 5980.50 | 5924.19 | 5931.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 5980.50 | 5924.19 | 5931.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 5933.50 | 5926.05 | 5931.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 5918.50 | 5927.30 | 5931.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 5908.50 | 5927.30 | 5931.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6197.00 | 5978.47 | 5953.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6197.00 | 5978.47 | 5953.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 09:15:00 | 6197.00 | 5978.47 | 5953.60 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 6005.00 | 6069.72 | 6069.88 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 6099.00 | 6075.58 | 6072.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 6124.50 | 6092.81 | 6081.78 | Break + close above crossover candle high |

### Cycle 58 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5994.50 | 6075.42 | 6075.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5960.50 | 6007.90 | 6038.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 5919.00 | 5913.09 | 5961.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 5927.50 | 5913.09 | 5961.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5964.50 | 5916.72 | 5950.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 5964.50 | 5916.72 | 5950.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 5975.00 | 5928.38 | 5952.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 5960.00 | 5928.38 | 5952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 5983.50 | 5948.30 | 5958.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:30:00 | 5995.50 | 5948.30 | 5958.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 5924.00 | 5949.95 | 5957.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 5911.00 | 5949.95 | 5957.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 5914.50 | 5887.65 | 5886.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 5914.50 | 5887.65 | 5886.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 5975.00 | 5911.82 | 5898.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5890.00 | 5907.46 | 5897.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 5890.00 | 5907.46 | 5897.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5861.50 | 5898.26 | 5894.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 5853.00 | 5898.26 | 5894.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 5892.00 | 5897.01 | 5893.96 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 5862.00 | 5890.01 | 5891.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 5840.00 | 5878.41 | 5885.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 5886.00 | 5873.46 | 5881.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 5886.00 | 5873.46 | 5881.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 5888.50 | 5876.47 | 5882.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 5894.50 | 5876.47 | 5882.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 5913.50 | 5883.87 | 5885.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 5913.50 | 5883.87 | 5885.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 5941.50 | 5895.40 | 5890.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 6037.00 | 5923.72 | 5903.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 5950.50 | 5962.60 | 5934.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 5950.50 | 5962.60 | 5934.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 5960.00 | 5962.08 | 5936.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 5957.50 | 5962.08 | 5936.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 5915.00 | 5951.93 | 5936.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 5915.00 | 5951.93 | 5936.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 5905.00 | 5942.54 | 5933.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 5896.00 | 5942.54 | 5933.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 5859.00 | 5925.84 | 5926.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 5821.00 | 5904.87 | 5917.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 5931.00 | 5891.43 | 5904.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 5931.00 | 5891.43 | 5904.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5950.00 | 5903.14 | 5909.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 5971.50 | 5903.14 | 5909.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 5968.50 | 5916.21 | 5914.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 6013.50 | 5982.35 | 5968.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 6501.50 | 6511.58 | 6374.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 6485.00 | 6511.58 | 6374.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 6390.00 | 6472.99 | 6421.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 6365.50 | 6472.99 | 6421.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 6365.00 | 6451.39 | 6416.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 6365.00 | 6451.39 | 6416.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 6429.50 | 6432.34 | 6417.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 6550.00 | 6432.34 | 6417.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 6749.00 | 6807.14 | 6809.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 6749.00 | 6807.14 | 6809.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 6729.00 | 6791.51 | 6802.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 6720.00 | 6661.82 | 6701.05 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 6761.00 | 6726.54 | 6722.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 6919.00 | 6766.07 | 6740.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 6811.50 | 6826.08 | 6786.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 6811.50 | 6826.08 | 6786.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 6770.00 | 6811.73 | 6786.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 6770.00 | 6811.73 | 6786.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 6780.00 | 6805.39 | 6786.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:45:00 | 6789.50 | 6789.18 | 6781.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:30:00 | 6800.50 | 6787.35 | 6781.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 6736.50 | 6777.18 | 6777.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 6736.50 | 6777.18 | 6777.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 6736.50 | 6777.18 | 6777.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 6639.00 | 6749.54 | 6764.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 6669.00 | 6667.44 | 6707.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 6669.00 | 6667.44 | 6707.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 6660.00 | 6656.61 | 6688.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 6641.00 | 6659.51 | 6684.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 6760.00 | 6687.49 | 6692.21 | SL hit (close>static) qty=1.00 sl=6726.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6744.00 | 6700.00 | 6697.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 6801.00 | 6743.29 | 6719.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 6765.00 | 6770.92 | 6740.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 6622.50 | 6770.92 | 6740.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6595.00 | 6735.74 | 6727.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 6592.00 | 6735.74 | 6727.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 6615.00 | 6711.59 | 6717.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 6584.50 | 6651.78 | 6685.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 6669.00 | 6635.57 | 6667.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 6664.50 | 6635.57 | 6667.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 6630.00 | 6634.46 | 6664.51 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 6807.00 | 6684.92 | 6677.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 7073.00 | 6811.63 | 6751.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7010.00 | 7018.10 | 6892.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 7010.00 | 7018.10 | 6892.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 6980.00 | 7012.84 | 6930.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 6925.00 | 7012.84 | 6930.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 6929.00 | 6987.30 | 6932.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 6990.50 | 6987.30 | 6932.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 7050.00 | 6999.84 | 6943.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 7246.00 | 7146.48 | 7058.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 7080.00 | 7183.94 | 7187.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 7080.00 | 7183.94 | 7187.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 6992.00 | 7145.55 | 7169.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 7190.00 | 7146.35 | 7165.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 7066.00 | 7132.00 | 7155.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 7056.00 | 7102.04 | 7136.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 6712.70 | 6927.50 | 7039.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 6703.20 | 6927.50 | 7039.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 6962.50 | 6835.48 | 6927.92 | SL hit (close>ema200) qty=0.50 sl=6835.48 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 6962.50 | 6835.48 | 6927.92 | SL hit (close>ema200) qty=0.50 sl=6835.48 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:00:00 | 7060.00 | 6903.67 | 6944.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 7415.00 | 7046.87 | 7004.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 7415.00 | 7046.87 | 7004.89 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 7043.50 | 7174.14 | 7186.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 6832.00 | 7046.02 | 7119.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 7000.00 | 6968.75 | 7060.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:15:00 | 7109.50 | 6968.75 | 7060.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 6960.00 | 6967.00 | 7051.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 6870.00 | 6967.00 | 7051.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 6947.00 | 6971.52 | 7025.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 6945.50 | 6982.82 | 7025.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 7270.50 | 7060.11 | 7041.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 7298.50 | 7107.79 | 7064.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 7146.00 | 7168.41 | 7125.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 7146.00 | 7168.41 | 7125.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 7150.00 | 7164.72 | 7127.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 7169.00 | 7164.72 | 7127.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 7160.50 | 7163.88 | 7130.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 7229.00 | 7154.65 | 7136.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 14:30:00 | 7235.00 | 7154.57 | 7142.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 7243.00 | 7159.66 | 7145.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 7214.00 | 7177.90 | 7156.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 7190.50 | 7187.30 | 7167.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 7160.00 | 7187.30 | 7167.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 7194.50 | 7188.74 | 7169.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 7178.00 | 7188.74 | 7169.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 7161.00 | 7183.19 | 7169.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 7336.00 | 7183.19 | 7169.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 7238.00 | 7208.85 | 7195.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 7218.50 | 7245.01 | 7247.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 7140.00 | 7221.83 | 7234.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7192.00 | 7101.24 | 7128.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 7192.00 | 7101.24 | 7128.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 7164.50 | 7113.89 | 7131.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 7206.50 | 7113.89 | 7131.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 7235.00 | 7153.65 | 7147.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 7274.00 | 7186.66 | 7164.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 7223.00 | 7237.72 | 7208.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 7223.00 | 7237.72 | 7208.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 7264.00 | 7242.98 | 7213.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 7271.50 | 7242.98 | 7213.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 7199.00 | 7248.43 | 7231.19 | SL hit (close<static) qty=1.00 sl=7211.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 7275.50 | 7240.24 | 7230.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 7208.50 | 7237.87 | 7232.08 | SL hit (close<static) qty=1.00 sl=7211.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 7209.00 | 7227.41 | 7228.51 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 7259.00 | 7233.73 | 7231.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 11:15:00 | 7272.50 | 7241.48 | 7235.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 7236.00 | 7242.71 | 7236.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 7236.50 | 7242.71 | 7236.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 7233.00 | 7240.77 | 7236.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 7230.00 | 7240.77 | 7236.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 7230.00 | 7238.61 | 7235.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 7262.00 | 7238.61 | 7235.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 7245.00 | 7239.89 | 7236.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-07 10:15:00 | 7988.20 | 7653.55 | 7549.66 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-07 10:15:00 | 7969.50 | 7653.55 | 7549.66 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 11:30:00 | 7228.50 | 2025-05-20 13:15:00 | 7106.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-20 12:30:00 | 7225.50 | 2025-05-20 13:15:00 | 7106.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-30 09:30:00 | 7616.50 | 2025-05-30 12:15:00 | 7480.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-06-09 10:15:00 | 7478.50 | 2025-06-13 09:15:00 | 7104.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 12:15:00 | 7472.50 | 2025-06-13 09:15:00 | 7098.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:00:00 | 7480.00 | 2025-06-13 09:15:00 | 7106.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:45:00 | 7472.50 | 2025-06-13 09:15:00 | 7098.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 10:45:00 | 7420.50 | 2025-06-16 09:15:00 | 7049.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 11:45:00 | 7435.00 | 2025-06-16 09:15:00 | 7063.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:15:00 | 7425.00 | 2025-06-16 09:15:00 | 7053.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:15:00 | 7478.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-06-09 12:15:00 | 7472.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-06-09 13:00:00 | 7480.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-06-09 13:45:00 | 7472.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-06-10 10:45:00 | 7420.50 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-06-11 11:45:00 | 7435.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2025-06-11 12:15:00 | 7425.00 | 2025-06-16 14:15:00 | 7140.00 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2025-07-03 13:30:00 | 6625.00 | 2025-07-03 14:15:00 | 6595.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-14 14:00:00 | 6844.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-15 09:15:00 | 6869.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-15 11:00:00 | 6844.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-15 13:00:00 | 6840.00 | 2025-07-16 13:15:00 | 6771.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-21 15:15:00 | 6675.00 | 2025-07-29 15:15:00 | 6570.00 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-07-24 10:00:00 | 6675.00 | 2025-07-29 15:15:00 | 6570.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-07-31 11:45:00 | 6615.00 | 2025-08-01 11:15:00 | 6561.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-01 10:15:00 | 6587.00 | 2025-08-01 11:15:00 | 6561.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-08 09:15:00 | 6401.00 | 2025-08-18 13:15:00 | 6276.50 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-08-21 10:15:00 | 6389.00 | 2025-08-21 15:15:00 | 6326.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-21 14:30:00 | 6399.00 | 2025-08-21 15:15:00 | 6326.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-25 11:30:00 | 6288.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-08-25 12:00:00 | 6279.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-25 13:45:00 | 6295.50 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-25 15:15:00 | 6222.00 | 2025-08-26 10:15:00 | 6400.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-03 14:45:00 | 6429.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-04 09:30:00 | 6460.50 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-09-05 11:30:00 | 6430.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-08 09:15:00 | 6435.00 | 2025-09-08 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-10 09:30:00 | 6489.00 | 2025-09-11 11:15:00 | 6442.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-10 11:30:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-10 13:30:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-10 14:00:00 | 6474.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-11 09:15:00 | 6511.00 | 2025-09-11 12:15:00 | 6437.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-09-26 11:15:00 | 6315.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2025-09-26 15:00:00 | 6306.50 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-09-29 09:15:00 | 6285.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-09-29 10:00:00 | 6299.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2025-10-01 12:15:00 | 6296.00 | 2025-10-09 13:15:00 | 6204.00 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-10-14 09:15:00 | 6220.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-14 10:15:00 | 6230.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-14 14:30:00 | 6206.50 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-15 09:30:00 | 6229.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-15 15:15:00 | 6210.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-16 15:15:00 | 6197.00 | 2025-10-17 10:15:00 | 6158.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-24 09:15:00 | 6089.00 | 2025-11-11 10:15:00 | 5784.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 12:30:00 | 6080.00 | 2025-11-11 10:15:00 | 5776.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 14:30:00 | 6082.50 | 2025-11-11 10:15:00 | 5778.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 09:45:00 | 6082.00 | 2025-11-11 10:15:00 | 5777.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 10:45:00 | 6078.00 | 2025-11-11 10:15:00 | 5774.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:00:00 | 6086.00 | 2025-11-11 10:15:00 | 5781.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:45:00 | 6085.00 | 2025-11-11 10:15:00 | 5780.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 13:15:00 | 6089.00 | 2025-11-11 10:15:00 | 5784.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:15:00 | 6070.00 | 2025-11-11 10:15:00 | 5766.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 6089.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-10-24 12:30:00 | 6080.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-10-24 14:30:00 | 6082.50 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-10-27 09:45:00 | 6082.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-10-28 10:45:00 | 6078.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-10-28 12:00:00 | 6086.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-10-28 12:45:00 | 6085.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-10-29 13:15:00 | 6089.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-10-29 14:15:00 | 6070.00 | 2025-11-12 12:15:00 | 5795.50 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-12-04 13:45:00 | 5799.00 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-04 14:30:00 | 5781.50 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-12-04 15:15:00 | 5787.00 | 2025-12-05 10:15:00 | 5941.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-12-11 09:15:00 | 5916.00 | 2025-12-17 10:15:00 | 5952.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-12-11 10:00:00 | 5899.50 | 2025-12-17 10:15:00 | 5952.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-12-22 11:15:00 | 5900.50 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-22 14:15:00 | 5910.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-22 15:15:00 | 5910.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-23 10:45:00 | 5910.50 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-24 10:30:00 | 5859.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-12-24 11:45:00 | 5855.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-24 13:15:00 | 5848.00 | 2025-12-26 12:15:00 | 5990.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-01-09 14:30:00 | 5918.50 | 2026-01-12 09:15:00 | 6197.00 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2026-01-09 15:00:00 | 5908.50 | 2026-01-12 09:15:00 | 6197.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-01-23 10:15:00 | 5911.00 | 2026-01-28 13:15:00 | 5914.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-02-13 09:15:00 | 6550.00 | 2026-02-23 09:15:00 | 6749.00 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2026-03-02 09:45:00 | 6789.50 | 2026-03-02 11:15:00 | 6736.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-03-02 10:30:00 | 6800.50 | 2026-03-02 11:15:00 | 6736.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-05 11:45:00 | 6641.00 | 2026-03-05 14:15:00 | 6760.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-03-17 11:30:00 | 7246.00 | 2026-03-19 13:15:00 | 7080.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-03-20 12:15:00 | 7066.00 | 2026-03-23 10:15:00 | 6712.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 7056.00 | 2026-03-23 10:15:00 | 6703.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 7066.00 | 2026-03-24 09:15:00 | 6962.50 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2026-03-20 13:45:00 | 7056.00 | 2026-03-24 09:15:00 | 6962.50 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2026-03-24 12:00:00 | 7060.00 | 2026-03-24 13:15:00 | 7415.00 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-04-01 10:15:00 | 6870.00 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2026-04-01 14:15:00 | 6947.00 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2026-04-01 15:15:00 | 6945.50 | 2026-04-02 13:15:00 | 7270.50 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-04-08 09:30:00 | 7229.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-08 14:30:00 | 7235.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-09 09:15:00 | 7243.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-09 10:30:00 | 7214.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-04-10 09:15:00 | 7336.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-04-13 10:00:00 | 7238.00 | 2026-04-16 14:15:00 | 7218.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-24 09:15:00 | 7271.50 | 2026-04-24 13:15:00 | 7199.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 7275.50 | 2026-04-27 11:15:00 | 7208.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-29 09:15:00 | 7262.00 | 2026-05-07 10:15:00 | 7988.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 10:00:00 | 7245.00 | 2026-05-07 10:15:00 | 7969.50 | TARGET_HIT | 1.00 | 10.00% |
