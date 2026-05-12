# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 8851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 0 |
| TARGET_HIT | 15 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 25
- **Target hits / Stop hits / Partials:** 15 / 25 / 0
- **Avg / median % per leg:** 1.24% / -2.76%
- **Sum % (uncompounded):** 49.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 15 | 53.6% | 15 | 13 | 0 | 3.69% | 103.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 15 | 53.6% | 15 | 13 | 0 | 3.69% | 103.4% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -4.48% | -53.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -4.48% | -53.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 15 | 37.5% | 15 | 25 | 0 | 1.24% | 49.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 2110.60 | 1895.97 | 1895.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 12:15:00 | 2130.80 | 1933.20 | 1915.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 15:15:00 | 2183.85 | 2190.09 | 2113.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:15:00 | 2191.50 | 2190.09 | 2113.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 2853.05 | 2928.73 | 2811.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:30:00 | 2822.00 | 2928.73 | 2811.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 2819.35 | 2926.75 | 2811.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:45:00 | 2820.50 | 2926.75 | 2811.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 2805.85 | 2925.54 | 2811.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:00:00 | 2805.85 | 2925.54 | 2811.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 2793.90 | 2924.24 | 2811.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:00:00 | 2793.90 | 2924.24 | 2811.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 3065.00 | 3141.64 | 3063.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:45:00 | 3069.90 | 3141.64 | 3063.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 3062.00 | 3140.85 | 3063.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 3034.00 | 3140.85 | 3063.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 3065.00 | 3140.09 | 3063.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 3073.00 | 3138.59 | 3063.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 15:00:00 | 3073.70 | 3136.87 | 3063.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:15:00 | 3075.65 | 3136.14 | 3063.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:15:00 | 3087.90 | 3124.87 | 3062.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-08 10:15:00 | 3380.30 | 3140.95 | 3086.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 15:15:00 | 3225.00 | 3623.80 | 3624.97 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 09:15:00 | 3773.40 | 3622.31 | 3622.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 11:15:00 | 3810.00 | 3625.75 | 3623.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 3640.00 | 3675.29 | 3652.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 3640.00 | 3675.29 | 3652.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 3640.00 | 3675.29 | 3652.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:15:00 | 3645.40 | 3675.29 | 3652.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:15:00 | 3655.50 | 3673.85 | 3651.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 11:00:00 | 3647.50 | 3671.15 | 3651.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 3663.80 | 3668.68 | 3650.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 3636.10 | 3668.15 | 3650.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:45:00 | 3636.20 | 3668.15 | 3650.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 3539.70 | 3664.23 | 3648.81 | SL hit (close<static) qty=1.00 sl=3553.25 alert=retest2 |

### Cycle 4 — SELL (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 11:15:00 | 5487.20 | 6480.22 | 6482.72 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 09:15:00 | 6944.30 | 6371.01 | 6370.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 7297.85 | 6540.23 | 6462.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 6625.00 | 6648.29 | 6532.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 13:00:00 | 6625.00 | 6648.29 | 6532.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 6604.35 | 6647.20 | 6533.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 6788.00 | 6567.30 | 6505.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 15:00:00 | 6650.00 | 6621.18 | 6544.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 6709.50 | 6621.30 | 6545.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 6684.00 | 6621.49 | 6545.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 6590.00 | 6622.79 | 6548.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 6455.00 | 6622.79 | 6548.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 6453.50 | 6621.10 | 6547.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 6453.50 | 6621.10 | 6547.97 | SL hit (close<static) qty=1.00 sl=6518.25 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 14:15:00 | 6104.00 | 6492.29 | 6492.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 6012.00 | 6483.57 | 6488.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 6350.00 | 6344.35 | 6406.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 6350.00 | 6344.35 | 6406.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 6496.00 | 6345.90 | 6405.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 6480.00 | 6345.90 | 6405.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 6480.50 | 6347.24 | 6406.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:45:00 | 6459.00 | 6349.96 | 6407.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 14:30:00 | 6458.50 | 6351.55 | 6407.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 6510.50 | 6349.74 | 6403.70 | SL hit (close>static) qty=1.00 sl=6510.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 6575.00 | 6433.30 | 6432.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 6626.50 | 6446.86 | 6440.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 6444.00 | 6459.40 | 6446.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 6462.00 | 6459.43 | 6446.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 6486.00 | 6458.52 | 6446.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 6472.50 | 6458.79 | 6446.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 09:15:00 | 7134.60 | 6601.30 | 6528.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 7327.50 | 7824.65 | 7825.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7708.54 | 7763.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 6176.00 | 6144.37 | 6514.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:15:00 | 6165.00 | 6144.37 | 6514.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 6580.50 | 6184.87 | 6503.42 | SL hit (close>static) qty=1.00 sl=6560.00 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 7815.00 | 6741.32 | 6739.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 7909.50 | 6804.35 | 6771.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 7312.50 | 7367.95 | 7122.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 12:00:00 | 7312.50 | 7367.95 | 7122.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 7009.50 | 7367.46 | 7144.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 6968.00 | 7367.46 | 7144.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 7088.50 | 7364.68 | 7144.43 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 6567.50 | 6990.22 | 6991.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6482.00 | 6962.72 | 6977.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 7048.00 | 6811.69 | 6892.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 6966.00 | 6813.23 | 6893.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 6903.00 | 6818.93 | 6894.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 7108.00 | 6822.77 | 6893.32 | SL hit (close>static) qty=1.00 sl=7065.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 7715.00 | 6955.85 | 6954.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 7874.00 | 6964.99 | 6959.11 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-21 12:15:00 | 3073.00 | 2024-01-08 10:15:00 | 3380.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 15:00:00 | 3073.70 | 2024-01-08 10:15:00 | 3381.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-22 09:15:00 | 3075.65 | 2024-01-08 10:15:00 | 3383.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-27 09:15:00 | 3087.90 | 2024-01-08 10:15:00 | 3396.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-12 15:00:00 | 3700.50 | 2024-02-13 09:15:00 | 3439.50 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest2 | 2024-02-13 15:15:00 | 3700.00 | 2024-02-14 09:15:00 | 3587.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-02-14 14:00:00 | 3729.90 | 2024-02-20 09:15:00 | 4102.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-28 14:45:00 | 3771.20 | 2024-02-29 09:15:00 | 3594.00 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2024-03-05 11:00:00 | 3773.00 | 2024-03-06 09:15:00 | 3668.85 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-04-15 10:15:00 | 3645.40 | 2024-04-19 09:15:00 | 3539.70 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-04-15 13:15:00 | 3655.50 | 2024-04-19 09:15:00 | 3539.70 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-04-16 11:00:00 | 3647.50 | 2024-04-19 09:15:00 | 3539.70 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-04-18 09:15:00 | 3663.80 | 2024-04-19 09:15:00 | 3539.70 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-04-22 09:15:00 | 3691.00 | 2024-05-06 09:15:00 | 4060.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-22 11:45:00 | 3692.00 | 2024-05-06 09:15:00 | 4061.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 09:15:00 | 3747.85 | 2024-05-06 09:15:00 | 4122.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-28 10:00:00 | 3676.80 | 2024-05-28 11:15:00 | 3623.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-06-06 12:00:00 | 3751.50 | 2024-06-12 15:15:00 | 4126.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 13:45:00 | 3751.95 | 2024-06-12 15:15:00 | 4127.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 09:45:00 | 3749.00 | 2024-06-12 15:15:00 | 4123.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 11:00:00 | 3756.75 | 2024-06-12 15:15:00 | 4132.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 09:15:00 | 3800.00 | 2024-06-12 15:15:00 | 4180.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-15 09:15:00 | 6788.00 | 2025-04-24 09:15:00 | 6453.50 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest2 | 2025-04-22 15:00:00 | 6650.00 | 2025-04-24 09:15:00 | 6453.50 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-04-23 09:15:00 | 6709.50 | 2025-04-24 09:15:00 | 6453.50 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-04-23 10:15:00 | 6684.00 | 2025-04-24 09:15:00 | 6453.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-05-16 12:45:00 | 6459.00 | 2025-05-20 10:15:00 | 6510.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-16 14:30:00 | 6458.50 | 2025-05-20 10:15:00 | 6510.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-20 11:30:00 | 6445.00 | 2025-05-21 09:15:00 | 6764.00 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2025-05-23 14:00:00 | 6408.50 | 2025-05-27 09:15:00 | 6580.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-06-04 09:30:00 | 6299.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-06-04 10:15:00 | 6302.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-06-05 14:15:00 | 6304.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-06-06 09:15:00 | 6276.00 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2025-06-09 10:15:00 | 6335.00 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2025-06-20 11:00:00 | 6486.00 | 2025-07-02 09:15:00 | 7134.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 14:15:00 | 6472.50 | 2025-07-02 09:15:00 | 7119.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-03 14:30:00 | 6176.00 | 2026-02-06 11:15:00 | 6580.50 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2026-02-03 15:15:00 | 6165.00 | 2026-02-06 11:15:00 | 6580.50 | STOP_HIT | 1.00 | -6.74% |
| SELL | retest2 | 2026-04-09 09:15:00 | 6903.00 | 2026-04-10 09:15:00 | 7108.00 | STOP_HIT | 1.00 | -2.97% |
