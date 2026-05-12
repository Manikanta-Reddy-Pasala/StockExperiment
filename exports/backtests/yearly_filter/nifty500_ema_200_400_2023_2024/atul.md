# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 7090.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 2 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 34
- **Target hits / Stop hits / Partials:** 7 / 36 / 5
- **Avg / median % per leg:** 0.77% / -1.30%
- **Sum % (uncompounded):** 36.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 3 | 21 | 0 | -0.11% | -2.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 3 | 21 | 0 | -0.11% | -2.6% |
| SELL (all) | 24 | 10 | 41.7% | 4 | 15 | 5 | 1.64% | 39.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 10 | 41.7% | 4 | 15 | 5 | 1.64% | 39.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 14 | 29.2% | 7 | 36 | 5 | 0.77% | 36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 7194.05 | 6846.07 | 6844.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 7226.50 | 6860.35 | 6851.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 7167.15 | 7179.34 | 7052.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 10:15:00 | 7147.30 | 7179.34 | 7052.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 7072.85 | 7173.54 | 7058.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 7072.85 | 7173.54 | 7058.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 7112.40 | 7172.93 | 7058.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 12:00:00 | 7126.25 | 7172.46 | 7058.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 13:15:00 | 7020.00 | 7163.99 | 7059.44 | SL hit (close<static) qty=1.00 sl=7049.95 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 6832.00 | 7008.20 | 7008.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 6806.30 | 7006.19 | 7007.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 6681.20 | 6651.10 | 6798.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-06 11:45:00 | 6670.80 | 6651.10 | 6798.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 6783.35 | 6621.96 | 6711.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 11:00:00 | 6783.35 | 6621.96 | 6711.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 11:15:00 | 6770.00 | 6623.43 | 6711.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 15:00:00 | 6750.30 | 6627.79 | 6712.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 09:15:00 | 6793.00 | 6630.88 | 6713.32 | SL hit (close>static) qty=1.00 sl=6791.15 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 11:15:00 | 7097.00 | 6767.72 | 6766.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 12:15:00 | 7156.55 | 6858.50 | 6817.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 6899.95 | 6944.84 | 6872.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 11:00:00 | 6899.95 | 6944.84 | 6872.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 6888.00 | 6944.27 | 6872.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 6870.85 | 6944.27 | 6872.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 6894.00 | 6943.18 | 6872.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:30:00 | 6874.70 | 6943.18 | 6872.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 6846.95 | 6942.22 | 6872.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 6846.95 | 6942.22 | 6872.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 6855.00 | 6941.35 | 6872.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 09:30:00 | 6825.00 | 6940.36 | 6872.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 6823.85 | 6939.20 | 6872.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:45:00 | 6811.55 | 6939.20 | 6872.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 6872.80 | 6923.36 | 6868.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:45:00 | 6870.00 | 6923.36 | 6868.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 6870.00 | 6922.83 | 6868.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:45:00 | 6872.00 | 6922.83 | 6868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 6853.65 | 6922.14 | 6868.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:00:00 | 6853.65 | 6922.14 | 6868.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 6840.60 | 6921.33 | 6868.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 6840.60 | 6921.33 | 6868.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 6854.05 | 6920.66 | 6868.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:15:00 | 6894.95 | 6919.67 | 6868.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 10:45:00 | 6894.90 | 6919.46 | 6868.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 6886.05 | 6917.55 | 6868.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 09:15:00 | 6822.25 | 6916.13 | 6868.29 | SL hit (close<static) qty=1.00 sl=6840.60 alert=retest2 |

### Cycle 4 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 6470.00 | 6830.88 | 6831.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 6376.60 | 6826.36 | 6829.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 11:15:00 | 6030.75 | 6029.63 | 6217.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-03 12:00:00 | 6030.75 | 6029.63 | 6217.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 6208.15 | 6013.15 | 6177.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:30:00 | 6219.00 | 6013.15 | 6177.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 6178.60 | 6014.80 | 6177.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 6002.00 | 6030.13 | 6178.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 09:15:00 | 6214.35 | 5983.52 | 6097.94 | SL hit (close>static) qty=1.00 sl=6207.25 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 6499.70 | 6030.47 | 6029.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 6505.85 | 6039.33 | 6033.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 7680.00 | 7821.01 | 7514.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 10:00:00 | 7680.00 | 7821.01 | 7514.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 7536.45 | 7774.40 | 7533.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:45:00 | 7538.00 | 7774.40 | 7533.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 7553.95 | 7772.20 | 7533.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:15:00 | 7590.00 | 7734.80 | 7531.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 7583.95 | 7731.61 | 7531.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 11:15:00 | 7579.00 | 7750.68 | 7574.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 12:15:00 | 7475.50 | 7745.88 | 7573.97 | SL hit (close<static) qty=1.00 sl=7529.40 alert=retest2 |

### Cycle 6 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 7250.25 | 7614.15 | 7615.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 7238.10 | 7610.41 | 7613.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 12:15:00 | 7470.35 | 7458.98 | 7519.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 12:15:00 | 7470.35 | 7458.98 | 7519.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 7470.35 | 7458.98 | 7519.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 7410.90 | 7463.74 | 7519.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 7040.35 | 7443.44 | 7506.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 6669.81 | 7087.02 | 7251.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 6985.50 | 6010.17 | 6009.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 7027.50 | 6020.30 | 6014.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 6998.00 | 7005.31 | 6742.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:00:00 | 6998.00 | 7005.31 | 6742.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7335.93 | 7098.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 6986.50 | 7335.93 | 7098.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 6949.00 | 7332.08 | 7098.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 6949.00 | 7332.08 | 7098.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 6673.00 | 6948.83 | 6949.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6571.00 | 6939.72 | 6944.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 6480.00 | 6478.32 | 6618.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:30:00 | 6477.00 | 6478.32 | 6618.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6227.00 | 5931.17 | 6099.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 6227.00 | 5931.17 | 6099.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 6220.00 | 5934.04 | 6099.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 6220.50 | 5934.04 | 6099.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 6150.50 | 5948.75 | 6101.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 6150.50 | 5948.75 | 6101.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 6103.00 | 5950.29 | 6101.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 6089.50 | 5958.99 | 6102.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 15:15:00 | 5785.02 | 5945.25 | 6061.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 5885.00 | 5883.99 | 6004.72 | SL hit (close>ema200) qty=0.50 sl=5883.99 alert=retest2 |

### Cycle 9 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 6165.00 | 6022.96 | 6022.34 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5821.00 | 6021.90 | 6022.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 5755.50 | 6019.25 | 6020.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 5974.00 | 5952.79 | 5983.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 6023.00 | 5953.49 | 5984.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 6023.00 | 5953.49 | 5984.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 6120.00 | 5955.14 | 5984.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:45:00 | 6131.50 | 5955.14 | 5984.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 5986.50 | 5996.11 | 6003.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 5940.00 | 5996.11 | 6003.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 6403.50 | 5996.20 | 6003.10 | SL hit (close>static) qty=1.00 sl=6032.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 6200.00 | 6010.32 | 6010.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 6310.00 | 6021.90 | 6015.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 6383.50 | 6405.66 | 6268.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 6383.50 | 6405.66 | 6268.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 6350.00 | 6403.10 | 6269.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 6307.50 | 6403.10 | 6269.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6278.50 | 6412.44 | 6285.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 6275.50 | 6412.44 | 6285.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 6317.50 | 6411.50 | 6285.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:45:00 | 6431.00 | 6344.55 | 6268.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 12:15:00 | 6249.00 | 6342.48 | 6269.09 | SL hit (close<static) qty=1.00 sl=6255.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-22 12:00:00 | 7126.25 | 2023-09-25 13:15:00 | 7020.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-12-04 15:00:00 | 6750.30 | 2023-12-05 09:15:00 | 6793.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-12-08 13:00:00 | 6735.00 | 2023-12-08 15:15:00 | 6809.95 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-12-08 14:45:00 | 6749.80 | 2023-12-08 15:15:00 | 6809.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-01-12 10:15:00 | 6894.95 | 2024-01-15 09:15:00 | 6822.25 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-01-12 10:45:00 | 6894.90 | 2024-01-15 09:15:00 | 6822.25 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-01-12 14:30:00 | 6886.05 | 2024-01-15 09:15:00 | 6822.25 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-01-15 14:45:00 | 6887.65 | 2024-01-16 11:15:00 | 6835.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-04-15 09:15:00 | 6002.00 | 2024-05-03 09:15:00 | 6214.35 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-05-03 11:00:00 | 6130.55 | 2024-05-03 11:15:00 | 6212.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-05-06 09:30:00 | 6118.65 | 2024-05-30 09:15:00 | 5812.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 10:00:00 | 6100.20 | 2024-05-30 09:15:00 | 5795.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:45:00 | 6065.00 | 2024-05-30 10:15:00 | 5761.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:30:00 | 6118.65 | 2024-06-04 11:15:00 | 5506.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-07 10:00:00 | 6100.20 | 2024-06-04 11:15:00 | 5490.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-09 09:45:00 | 6065.00 | 2024-06-04 11:15:00 | 5458.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-11 11:30:00 | 6059.15 | 2024-06-12 12:15:00 | 6162.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-06-12 09:15:00 | 6066.15 | 2024-06-12 12:15:00 | 6162.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-06-12 10:15:00 | 6070.00 | 2024-06-12 12:15:00 | 6162.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-09-26 15:15:00 | 7590.00 | 2024-10-07 12:15:00 | 7475.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-27 10:15:00 | 7583.95 | 2024-10-07 12:15:00 | 7475.50 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-10-07 11:15:00 | 7579.00 | 2024-10-07 12:15:00 | 7475.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-10-08 09:15:00 | 7570.00 | 2024-10-21 15:15:00 | 7580.60 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-10-21 13:30:00 | 7640.75 | 2024-10-22 10:15:00 | 7570.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-10-22 09:15:00 | 7645.40 | 2024-10-22 12:15:00 | 7492.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-10-23 10:45:00 | 7652.25 | 2024-10-25 09:15:00 | 7525.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-23 11:15:00 | 7629.10 | 2024-10-25 09:15:00 | 7525.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-10-24 15:15:00 | 7680.00 | 2024-10-25 09:15:00 | 7525.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-10-30 10:15:00 | 7674.95 | 2024-11-11 12:15:00 | 7573.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-10-30 12:30:00 | 7684.95 | 2024-11-11 12:15:00 | 7573.40 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-10-30 13:45:00 | 7683.55 | 2024-11-11 12:15:00 | 7573.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-12-12 09:15:00 | 7410.90 | 2024-12-13 10:15:00 | 7040.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 7410.90 | 2025-01-13 12:15:00 | 6669.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 6089.50 | 2025-12-01 15:15:00 | 5785.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 6089.50 | 2025-12-10 09:15:00 | 5885.00 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-12-19 10:15:00 | 6090.50 | 2025-12-29 13:15:00 | 6160.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-23 15:15:00 | 6051.00 | 2025-12-29 13:15:00 | 6160.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-12-30 12:00:00 | 6072.00 | 2025-12-31 15:15:00 | 6151.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5986.50 | 2026-01-12 12:15:00 | 6078.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-12 12:45:00 | 6001.00 | 2026-01-12 13:15:00 | 6184.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-02-02 10:15:00 | 5940.00 | 2026-02-03 09:15:00 | 6403.50 | STOP_HIT | 1.00 | -7.80% |
| BUY | retest2 | 2026-03-13 14:45:00 | 6431.00 | 2026-03-16 12:15:00 | 6249.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-18 09:45:00 | 6360.00 | 2026-03-19 13:15:00 | 6247.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-03-30 14:45:00 | 6362.00 | 2026-04-06 09:15:00 | 6160.50 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-04-01 09:45:00 | 6367.50 | 2026-04-06 09:15:00 | 6160.50 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-04-09 12:30:00 | 6483.50 | 2026-05-06 13:15:00 | 7071.90 | TARGET_HIT | 1.00 | 9.08% |
| BUY | retest2 | 2026-04-10 09:15:00 | 6501.00 | 2026-05-06 14:15:00 | 7131.85 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2026-04-15 10:00:00 | 6429.00 | 2026-05-07 09:15:00 | 7151.10 | TARGET_HIT | 1.00 | 11.23% |
