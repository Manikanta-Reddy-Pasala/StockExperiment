# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 8188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 163 |
| ALERT1 | 101 |
| ALERT2 | 97 |
| ALERT2_SKIP | 47 |
| ALERT3 | 225 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 104 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 73
- **Target hits / Stop hits / Partials:** 6 / 99 / 8
- **Avg / median % per leg:** 0.18% / -0.99%
- **Sum % (uncompounded):** 20.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 8 | 20.0% | 6 | 34 | 0 | 0.15% | 5.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 8 | 20.0% | 6 | 34 | 0 | 0.15% | 5.9% |
| SELL (all) | 73 | 32 | 43.8% | 0 | 65 | 8 | 0.20% | 14.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.68% | -4.7% |
| SELL @ 3rd Alert (retest2) | 72 | 32 | 44.4% | 0 | 64 | 8 | 0.27% | 19.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.68% | -4.7% |
| retest2 (combined) | 112 | 40 | 35.7% | 6 | 98 | 8 | 0.23% | 25.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 6998.30 | 6871.30 | 6868.23 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 10:15:00 | 6790.45 | 6868.17 | 6873.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 6720.65 | 6808.55 | 6839.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 11:15:00 | 6795.85 | 6793.44 | 6826.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:00:00 | 6795.85 | 6793.44 | 6826.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 6776.50 | 6766.20 | 6798.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:45:00 | 6740.15 | 6759.62 | 6792.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 15:15:00 | 6830.00 | 6793.34 | 6798.24 | SL hit (close>static) qty=1.00 sl=6825.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 6838.90 | 6805.65 | 6802.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 14:15:00 | 6881.60 | 6827.62 | 6814.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 6834.05 | 6849.59 | 6829.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 11:15:00 | 6834.05 | 6849.59 | 6829.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 6834.05 | 6849.59 | 6829.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 6834.05 | 6849.59 | 6829.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 6807.00 | 6841.07 | 6827.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 6781.45 | 6841.07 | 6827.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 6793.00 | 6831.46 | 6824.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 6780.65 | 6831.46 | 6824.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 6800.80 | 6825.33 | 6822.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 6780.75 | 6825.33 | 6822.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 6812.95 | 6824.22 | 6822.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 6812.95 | 6824.22 | 6822.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 6804.40 | 6820.25 | 6820.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 6775.00 | 6810.67 | 6816.18 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 6897.10 | 6827.95 | 6823.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 10:15:00 | 6993.60 | 6861.08 | 6839.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 11:15:00 | 6845.10 | 6893.89 | 6874.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 6845.10 | 6893.89 | 6874.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 6845.10 | 6893.89 | 6874.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 6845.10 | 6893.89 | 6874.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 6908.35 | 6896.78 | 6877.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 13:15:00 | 6931.85 | 6896.78 | 6877.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:45:00 | 6932.90 | 6932.98 | 6916.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 7088.00 | 6914.79 | 6909.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 6884.90 | 6968.92 | 6977.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 6884.90 | 6968.92 | 6977.61 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 7043.00 | 6978.10 | 6976.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 10:15:00 | 7105.00 | 7060.20 | 7026.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 15:15:00 | 7100.00 | 7140.12 | 7108.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 7100.00 | 7140.12 | 7108.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 7100.00 | 7140.12 | 7108.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:45:00 | 7219.00 | 7148.22 | 7120.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 7184.95 | 7183.22 | 7150.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 6950.40 | 7136.66 | 7132.35 | SL hit (close<static) qty=1.00 sl=7070.25 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 6842.50 | 7077.83 | 7106.00 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 7494.70 | 7181.00 | 7145.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 7632.30 | 7448.64 | 7321.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 13:15:00 | 7454.30 | 7513.13 | 7399.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:00:00 | 7454.30 | 7513.13 | 7399.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 7830.00 | 7830.79 | 7762.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:00:00 | 7899.50 | 7844.53 | 7774.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:30:00 | 7862.50 | 7851.32 | 7790.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 7867.65 | 7820.03 | 7797.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 12:15:00 | 7729.00 | 7778.43 | 7781.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 7729.00 | 7778.43 | 7781.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 7701.85 | 7755.84 | 7770.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 7788.15 | 7758.49 | 7768.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 7788.15 | 7758.49 | 7768.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 7788.15 | 7758.49 | 7768.92 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 11:15:00 | 7840.80 | 7780.97 | 7777.70 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 7685.70 | 7769.49 | 7777.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 7651.55 | 7736.24 | 7760.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 7505.35 | 7494.43 | 7584.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:30:00 | 7498.00 | 7494.43 | 7584.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 7532.95 | 7515.69 | 7572.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:30:00 | 7501.40 | 7521.86 | 7565.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 7595.75 | 7538.10 | 7565.54 | SL hit (close>static) qty=1.00 sl=7577.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 7354.60 | 7277.83 | 7268.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 13:15:00 | 7405.70 | 7303.40 | 7281.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 7329.95 | 7361.82 | 7336.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 7329.95 | 7361.82 | 7336.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 7329.95 | 7361.82 | 7336.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 7329.95 | 7361.82 | 7336.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 7347.55 | 7358.97 | 7337.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 7316.50 | 7358.97 | 7337.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 7424.90 | 7372.16 | 7345.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:30:00 | 7363.95 | 7372.16 | 7345.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 7399.00 | 7391.75 | 7365.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 7355.10 | 7391.75 | 7365.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 7366.65 | 7386.73 | 7365.24 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 7321.70 | 7361.04 | 7363.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 7560.00 | 7398.20 | 7378.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 14:15:00 | 7644.95 | 7491.23 | 7435.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 7800.00 | 7872.00 | 7763.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 7800.00 | 7872.00 | 7763.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 7761.00 | 7849.80 | 7762.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 7761.00 | 7849.80 | 7762.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 7734.85 | 7826.81 | 7760.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 7724.65 | 7826.81 | 7760.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 7771.45 | 7815.74 | 7761.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 13:15:00 | 7799.95 | 7815.74 | 7761.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 7695.00 | 7780.75 | 7754.23 | SL hit (close<static) qty=1.00 sl=7711.45 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 10:15:00 | 7688.05 | 7736.57 | 7738.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 12:15:00 | 7654.00 | 7708.54 | 7724.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 7728.00 | 7696.55 | 7713.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 15:15:00 | 7728.00 | 7696.55 | 7713.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 7728.00 | 7696.55 | 7713.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 7630.70 | 7681.89 | 7705.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 7740.10 | 7693.53 | 7708.49 | SL hit (close>static) qty=1.00 sl=7735.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 7856.20 | 7726.06 | 7721.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 7896.35 | 7794.09 | 7759.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 15:15:00 | 7800.00 | 7820.34 | 7785.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 15:15:00 | 7800.00 | 7820.34 | 7785.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 7800.00 | 7820.34 | 7785.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 7780.45 | 7820.34 | 7785.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 7766.15 | 7809.50 | 7783.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 7766.15 | 7809.50 | 7783.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 7750.00 | 7797.60 | 7780.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:15:00 | 7789.50 | 7776.17 | 7773.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 15:15:00 | 7815.15 | 7835.74 | 7836.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 7815.15 | 7835.74 | 7836.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 7785.40 | 7825.67 | 7832.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 7956.05 | 7817.63 | 7821.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 14:15:00 | 7956.05 | 7817.63 | 7821.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 7956.05 | 7817.63 | 7821.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 7956.05 | 7817.63 | 7821.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 15:15:00 | 7970.00 | 7848.10 | 7835.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 09:15:00 | 8100.40 | 7898.56 | 7859.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 14:15:00 | 7972.75 | 8005.84 | 7937.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 15:00:00 | 7972.75 | 8005.84 | 7937.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 7966.55 | 7997.99 | 7940.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 7772.05 | 7997.99 | 7940.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 7888.00 | 7975.99 | 7935.48 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 7706.95 | 7885.13 | 7898.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 7647.75 | 7837.66 | 7876.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 7738.80 | 7730.55 | 7786.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 13:00:00 | 7738.80 | 7730.55 | 7786.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 7760.95 | 7711.36 | 7752.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 7735.00 | 7711.36 | 7752.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 7889.95 | 7747.08 | 7764.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 7864.05 | 7747.08 | 7764.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 7855.00 | 7768.66 | 7772.94 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 7834.55 | 7781.84 | 7778.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 7895.00 | 7814.25 | 7794.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 8015.70 | 8022.81 | 7950.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:45:00 | 8001.00 | 8022.81 | 7950.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 8012.20 | 8019.16 | 7961.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:45:00 | 7969.30 | 8019.16 | 7961.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 8000.00 | 8015.33 | 7965.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 7956.20 | 8015.33 | 7965.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 7964.95 | 8005.26 | 7965.10 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 7891.25 | 7946.52 | 7946.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 7792.50 | 7898.12 | 7922.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 7739.40 | 7713.59 | 7784.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 7739.40 | 7713.59 | 7784.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 7759.80 | 7725.26 | 7771.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 7830.75 | 7725.26 | 7771.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 7882.10 | 7756.62 | 7781.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 7882.10 | 7756.62 | 7781.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 7910.50 | 7787.40 | 7793.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 7910.50 | 7787.40 | 7793.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 7922.00 | 7814.32 | 7805.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 7973.50 | 7846.16 | 7820.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 7945.95 | 7954.47 | 7915.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 7945.95 | 7954.47 | 7915.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 7892.05 | 7941.99 | 7913.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 7892.05 | 7941.99 | 7913.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 7986.20 | 7950.83 | 7919.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 13:45:00 | 8001.00 | 7959.23 | 7926.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 8049.95 | 7977.37 | 7937.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 09:15:00 | 8801.10 | 8500.44 | 8375.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 9073.35 | 9104.60 | 9104.90 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 9149.95 | 9110.89 | 9107.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 9150.75 | 9120.64 | 9112.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 10:15:00 | 9079.05 | 9112.32 | 9109.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 10:15:00 | 9079.05 | 9112.32 | 9109.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 9079.05 | 9112.32 | 9109.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 9064.75 | 9112.32 | 9109.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 11:15:00 | 9055.05 | 9100.87 | 9104.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 13:15:00 | 9038.20 | 9080.92 | 9094.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 8922.90 | 8917.08 | 8964.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 13:45:00 | 8922.05 | 8917.08 | 8964.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 8922.20 | 8909.05 | 8948.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 8832.45 | 8901.46 | 8912.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 9011.20 | 8922.23 | 8917.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 9011.20 | 8922.23 | 8917.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 9247.00 | 9027.28 | 8976.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 9022.70 | 9043.11 | 8998.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 13:00:00 | 9022.70 | 9043.11 | 8998.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 9027.10 | 9035.41 | 9002.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 9027.10 | 9035.41 | 9002.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 8952.00 | 9018.73 | 8997.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 9073.25 | 9018.73 | 8997.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 13:15:00 | 8880.05 | 8978.17 | 8984.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 13:15:00 | 8880.05 | 8978.17 | 8984.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 15:15:00 | 8874.00 | 8965.91 | 8978.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 8567.65 | 8548.20 | 8624.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 8567.65 | 8548.20 | 8624.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 8534.95 | 8507.34 | 8543.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 8534.95 | 8507.34 | 8543.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 8468.00 | 8499.47 | 8536.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 14:45:00 | 8430.05 | 8500.46 | 8514.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 8365.45 | 8490.13 | 8508.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:00:00 | 8434.00 | 8476.96 | 8499.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:30:00 | 8431.65 | 8483.02 | 8492.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 8530.00 | 8492.42 | 8495.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 8483.65 | 8492.42 | 8495.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 8567.40 | 8504.80 | 8500.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 8567.40 | 8504.80 | 8500.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-07 11:15:00 | 8684.30 | 8540.70 | 8517.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 8954.05 | 8982.95 | 8901.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 8954.05 | 8982.95 | 8901.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 8899.95 | 8953.88 | 8907.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 8899.95 | 8953.88 | 8907.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 8864.95 | 8936.09 | 8903.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:00:00 | 8864.95 | 8936.09 | 8903.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 8929.45 | 8928.55 | 8905.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:30:00 | 8888.55 | 8928.55 | 8905.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 9041.40 | 8951.12 | 8917.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:45:00 | 8944.55 | 8951.12 | 8917.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 9090.00 | 9091.81 | 9050.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 9138.75 | 9092.45 | 9054.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 8982.10 | 9070.15 | 9051.12 | SL hit (close<static) qty=1.00 sl=9002.50 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 8989.75 | 9039.40 | 9039.54 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 9146.80 | 9060.88 | 9049.29 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 8998.70 | 9040.41 | 9042.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 8960.70 | 9024.47 | 9035.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 8838.05 | 8827.22 | 8897.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 8838.05 | 8827.22 | 8897.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 8838.05 | 8827.22 | 8897.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 8723.85 | 8802.88 | 8874.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 8687.00 | 8779.12 | 8857.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 8287.66 | 8528.60 | 8665.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 8252.65 | 8479.54 | 8617.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 8478.90 | 8475.97 | 8592.09 | SL hit (close>ema200) qty=0.50 sl=8475.97 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 8703.00 | 8335.72 | 8331.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 9297.65 | 8528.10 | 8419.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 9599.95 | 9637.59 | 9349.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:30:00 | 9663.75 | 9637.59 | 9349.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 10237.50 | 10391.57 | 10269.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 10237.50 | 10391.57 | 10269.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 10215.55 | 10356.37 | 10264.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 10215.55 | 10356.37 | 10264.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 10228.00 | 10330.69 | 10261.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 10103.85 | 10330.69 | 10261.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 10038.20 | 10221.07 | 10220.30 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 10022.20 | 10181.30 | 10202.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 9991.70 | 10143.38 | 10183.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 9445.45 | 9435.00 | 9564.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 14:00:00 | 9445.45 | 9435.00 | 9564.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 9550.00 | 9468.85 | 9557.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 9814.15 | 9468.85 | 9557.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 9722.70 | 9519.62 | 9572.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:00:00 | 9583.05 | 9581.77 | 9590.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 9628.20 | 9351.57 | 9336.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 9628.20 | 9351.57 | 9336.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 9924.00 | 9466.06 | 9389.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 10066.25 | 10069.02 | 9822.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 10066.25 | 10069.02 | 9822.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 10018.00 | 10053.43 | 9910.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:45:00 | 10120.00 | 10110.71 | 9949.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 9751.60 | 9951.47 | 9938.43 | SL hit (close<static) qty=1.00 sl=9851.05 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 9827.05 | 9926.58 | 9928.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 9625.00 | 9773.22 | 9840.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 9761.85 | 9734.75 | 9796.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 13:45:00 | 9759.75 | 9734.75 | 9796.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 9789.60 | 9745.72 | 9796.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 9789.60 | 9745.72 | 9796.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 9868.65 | 9770.30 | 9802.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 9900.00 | 9796.20 | 9811.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 9814.10 | 9799.78 | 9811.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:00:00 | 9740.00 | 9792.53 | 9805.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:00:00 | 9775.00 | 9795.64 | 9802.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 13:15:00 | 9777.45 | 9792.29 | 9800.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 13:45:00 | 9777.85 | 9788.15 | 9797.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 9779.35 | 9786.39 | 9795.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 9779.35 | 9786.39 | 9795.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 9762.00 | 9781.51 | 9792.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 9864.25 | 9781.51 | 9792.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 9843.80 | 9793.97 | 9797.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 9856.20 | 9806.42 | 9802.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 9856.20 | 9806.42 | 9802.83 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 9788.50 | 9800.11 | 9800.77 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 10042.00 | 9848.47 | 9822.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 11:15:00 | 10207.35 | 9953.36 | 9876.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 9997.10 | 10032.62 | 9939.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 15:00:00 | 9997.10 | 10032.62 | 9939.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 9991.10 | 10004.24 | 9967.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:45:00 | 9998.00 | 10004.24 | 9967.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 9951.05 | 9993.60 | 9966.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 9941.00 | 9993.60 | 9966.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 9959.45 | 9986.77 | 9965.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:15:00 | 9891.00 | 9986.77 | 9965.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 9864.00 | 9962.21 | 9956.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 9835.60 | 9962.21 | 9956.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 9824.90 | 9934.75 | 9944.45 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 10095.00 | 9939.54 | 9934.68 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 9887.15 | 9957.55 | 9967.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 9827.50 | 9931.54 | 9954.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 9859.90 | 9725.42 | 9790.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 9859.90 | 9725.42 | 9790.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 9859.90 | 9725.42 | 9790.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 9859.90 | 9725.42 | 9790.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 9845.00 | 9749.34 | 9795.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 9857.65 | 9749.34 | 9795.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 9789.35 | 9765.34 | 9795.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:45:00 | 9779.80 | 9765.34 | 9795.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 9790.00 | 9770.27 | 9794.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 9792.30 | 9770.27 | 9794.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 9799.95 | 9776.21 | 9795.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 9799.95 | 9776.21 | 9795.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 9750.00 | 9770.97 | 9791.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 9749.00 | 9770.97 | 9791.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 9715.85 | 9759.94 | 9784.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:30:00 | 9679.25 | 9731.09 | 9764.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 14:15:00 | 9784.90 | 9768.40 | 9767.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 14:15:00 | 9784.90 | 9768.40 | 9767.51 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 9751.35 | 9764.99 | 9766.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 9663.50 | 9744.69 | 9756.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 9264.00 | 9189.03 | 9345.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 11:15:00 | 9341.00 | 9239.42 | 9342.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 9341.00 | 9239.42 | 9342.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:45:00 | 9420.00 | 9239.42 | 9342.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 9335.90 | 9258.71 | 9341.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:45:00 | 9354.90 | 9258.71 | 9341.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 9357.00 | 9278.37 | 9342.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:45:00 | 9348.20 | 9278.37 | 9342.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 9591.40 | 9340.98 | 9365.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 9591.40 | 9340.98 | 9365.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 9501.00 | 9372.98 | 9377.84 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 9469.70 | 9392.33 | 9386.19 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 9288.10 | 9377.38 | 9388.62 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 9800.00 | 9435.39 | 9404.72 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 9562.80 | 9753.01 | 9757.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 9520.20 | 9706.45 | 9735.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 9816.25 | 9728.41 | 9743.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 9816.25 | 9728.41 | 9743.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 9816.25 | 9728.41 | 9743.27 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 9902.45 | 9778.04 | 9761.60 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 15:15:00 | 9736.00 | 9769.73 | 9769.98 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 9908.10 | 9797.40 | 9782.53 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 9701.15 | 9778.26 | 9782.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 9600.00 | 9700.32 | 9741.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 9700.00 | 9674.96 | 9713.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 9656.10 | 9671.19 | 9708.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 9656.10 | 9671.19 | 9708.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 9735.00 | 9671.19 | 9708.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 9599.95 | 9580.87 | 9640.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 9385.00 | 9538.22 | 9594.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 9410.30 | 9384.92 | 9382.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 14:15:00 | 9410.30 | 9384.92 | 9382.20 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 9326.10 | 9378.96 | 9380.76 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 9421.00 | 9376.75 | 9375.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 9526.75 | 9414.47 | 9393.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 9459.85 | 9472.40 | 9435.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 9459.85 | 9472.40 | 9435.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 9459.85 | 9472.40 | 9435.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 9459.85 | 9472.40 | 9435.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 9427.55 | 9463.43 | 9434.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 9427.55 | 9463.43 | 9434.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 9528.00 | 9476.34 | 9443.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:15:00 | 9537.15 | 9476.34 | 9443.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 13:00:00 | 9544.50 | 9489.98 | 9452.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 9389.50 | 9475.56 | 9452.64 | SL hit (close<static) qty=1.00 sl=9403.85 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 9075.00 | 9371.55 | 9407.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 8987.20 | 9294.68 | 9369.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 10:15:00 | 8735.30 | 8735.14 | 8905.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 11:00:00 | 8735.30 | 8735.14 | 8905.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 8864.95 | 8735.39 | 8826.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 8864.95 | 8735.39 | 8826.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 8770.15 | 8742.34 | 8821.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:45:00 | 8650.55 | 8736.10 | 8811.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 8696.00 | 8736.28 | 8804.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 14:15:00 | 8880.15 | 8684.20 | 8683.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 8880.15 | 8684.20 | 8683.24 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 8645.00 | 8676.36 | 8679.77 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 8932.25 | 8727.54 | 8702.72 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 8606.60 | 8686.16 | 8688.94 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 15:15:00 | 8710.95 | 8689.96 | 8689.53 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 8594.00 | 8681.33 | 8686.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 8530.00 | 8622.43 | 8655.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 8620.00 | 8554.80 | 8597.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 12:15:00 | 8620.00 | 8554.80 | 8597.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 8620.00 | 8554.80 | 8597.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:30:00 | 8651.10 | 8554.80 | 8597.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 8719.00 | 8587.64 | 8608.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:30:00 | 8700.00 | 8587.64 | 8608.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 8700.00 | 8610.11 | 8617.22 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 15:15:00 | 8686.50 | 8625.39 | 8623.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 8816.10 | 8663.53 | 8641.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 15:15:00 | 8602.10 | 8807.01 | 8744.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 15:15:00 | 8602.10 | 8807.01 | 8744.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 8602.10 | 8807.01 | 8744.21 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 8751.00 | 8803.41 | 8803.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 8600.00 | 8762.73 | 8785.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 7614.00 | 7522.79 | 7686.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 10:00:00 | 7614.00 | 7522.79 | 7686.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 8265.00 | 7671.23 | 7738.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 8265.00 | 7671.23 | 7738.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 11:15:00 | 8328.35 | 7802.66 | 7792.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 12:15:00 | 8568.20 | 7955.76 | 7862.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 15:15:00 | 8054.00 | 8068.18 | 7946.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 09:15:00 | 8420.65 | 8068.18 | 7946.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 8415.00 | 8447.68 | 8271.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:00:00 | 8611.25 | 8480.39 | 8302.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 12:00:00 | 8535.00 | 8491.32 | 8323.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 8201.00 | 8299.81 | 8311.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 8201.00 | 8299.81 | 8311.86 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 8645.00 | 8368.85 | 8342.15 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 8257.25 | 8359.83 | 8359.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 8148.00 | 8274.33 | 8314.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 10:15:00 | 8286.90 | 8270.84 | 8305.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 11:15:00 | 8247.85 | 8270.84 | 8305.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 8091.60 | 8234.99 | 8286.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 8053.75 | 8191.90 | 8245.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 8018.25 | 7896.54 | 7893.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 8018.25 | 7896.54 | 7893.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 15:15:00 | 8037.00 | 7924.63 | 7906.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 8009.75 | 8038.82 | 7992.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 8009.75 | 8038.82 | 7992.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 8009.75 | 8038.82 | 7992.51 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 7856.35 | 7950.70 | 7962.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 15:15:00 | 7848.00 | 7930.16 | 7952.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 7955.00 | 7935.13 | 7952.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 7955.00 | 7935.13 | 7952.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 7955.00 | 7935.13 | 7952.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 7948.10 | 7935.13 | 7952.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 7873.70 | 7922.84 | 7945.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 12:30:00 | 7856.55 | 7929.22 | 7943.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 15:15:00 | 7860.00 | 7918.15 | 7936.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 7986.95 | 7911.90 | 7918.70 | SL hit (close>static) qty=1.00 sl=7958.85 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 8003.85 | 7930.29 | 7926.44 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 7890.60 | 7922.35 | 7923.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 7829.15 | 7900.36 | 7912.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 7888.15 | 7887.14 | 7903.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:00:00 | 7888.15 | 7887.14 | 7903.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 7895.00 | 7888.71 | 7903.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 7882.30 | 7888.71 | 7903.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 7801.55 | 7871.28 | 7893.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 7775.00 | 7871.28 | 7893.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 7864.55 | 7869.93 | 7891.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 7919.75 | 7869.93 | 7891.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 7907.30 | 7877.41 | 7892.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 7838.85 | 7870.15 | 7888.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 7833.80 | 7862.88 | 7883.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 8053.15 | 7871.56 | 7877.46 | SL hit (close>static) qty=1.00 sl=7932.70 alert=retest2 |

### Cycle 73 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 8129.70 | 7923.19 | 7900.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 8154.35 | 8033.58 | 7966.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 8114.70 | 8158.00 | 8088.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 09:30:00 | 8114.00 | 8158.00 | 8088.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 8075.05 | 8130.20 | 8101.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 8075.05 | 8130.20 | 8101.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 8080.00 | 8120.16 | 8099.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 8263.70 | 8120.16 | 8099.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 8071.05 | 8157.90 | 8158.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 8071.05 | 8157.90 | 8158.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 10:15:00 | 8049.70 | 8116.69 | 8137.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 13:15:00 | 8094.95 | 8094.19 | 8120.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 14:00:00 | 8094.95 | 8094.19 | 8120.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 7996.45 | 8061.98 | 8098.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 7934.05 | 8032.60 | 8078.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 7940.00 | 7990.10 | 8035.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 7965.70 | 7990.10 | 8035.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:00:00 | 7969.90 | 7930.63 | 7974.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 8100.70 | 7964.65 | 7985.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 8100.70 | 7964.65 | 7985.91 | SL hit (close>static) qty=1.00 sl=8100.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 8069.45 | 8002.72 | 7999.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 8182.95 | 8050.38 | 8022.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 8040.00 | 8064.47 | 8039.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 13:15:00 | 8040.00 | 8064.47 | 8039.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 8040.00 | 8064.47 | 8039.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 8040.00 | 8064.47 | 8039.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 8025.00 | 8056.57 | 8038.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:30:00 | 8033.85 | 8056.57 | 8038.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 8025.00 | 8050.26 | 8037.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 8049.75 | 8050.26 | 8037.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 8006.35 | 8040.00 | 8034.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:30:00 | 8007.50 | 8040.00 | 8034.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 8018.55 | 8035.71 | 8033.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 8006.80 | 8035.71 | 8033.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 8016.60 | 8029.89 | 8030.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 7989.95 | 8017.91 | 8025.07 | Break + close below crossover candle low |

### Cycle 77 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 8097.65 | 8033.85 | 8031.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 8125.95 | 8054.91 | 8042.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 11:15:00 | 8044.35 | 8066.15 | 8054.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 8044.35 | 8066.15 | 8054.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 8044.35 | 8066.15 | 8054.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 8044.35 | 8066.15 | 8054.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 8021.00 | 8057.12 | 8051.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 8021.00 | 8057.12 | 8051.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 8056.85 | 8057.07 | 8052.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 8079.50 | 8061.55 | 8054.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 7952.15 | 8039.42 | 8045.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 7952.15 | 8039.42 | 8045.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 7650.30 | 7912.81 | 7974.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 7798.90 | 7777.49 | 7861.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 7853.45 | 7777.49 | 7861.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 7792.20 | 7780.43 | 7855.22 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 8030.10 | 7902.80 | 7890.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 8059.85 | 7968.92 | 7936.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 7996.90 | 8018.52 | 7971.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 7996.90 | 8018.52 | 7971.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 7985.25 | 8011.86 | 7972.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 7985.25 | 8011.86 | 7972.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 7882.00 | 7985.89 | 7964.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 8064.50 | 7985.89 | 7964.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 7941.00 | 7964.78 | 7966.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 11:15:00 | 7941.00 | 7964.78 | 7966.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 12:15:00 | 7891.50 | 7950.13 | 7959.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 10:15:00 | 7832.00 | 7831.70 | 7868.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-21 10:30:00 | 7830.50 | 7831.70 | 7868.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 7847.50 | 7834.86 | 7866.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:45:00 | 7869.50 | 7834.86 | 7866.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 7840.00 | 7835.89 | 7863.90 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 8108.00 | 7907.13 | 7889.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 8232.00 | 7972.11 | 7920.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 13:15:00 | 8243.00 | 8260.64 | 8177.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 14:00:00 | 8243.00 | 8260.64 | 8177.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 8118.50 | 8245.47 | 8192.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 8118.50 | 8245.47 | 8192.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 8089.50 | 8214.27 | 8183.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 8089.50 | 8214.27 | 8183.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 8185.00 | 8185.01 | 8175.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:30:00 | 8194.00 | 8185.01 | 8175.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 8159.50 | 8179.91 | 8173.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 8159.50 | 8179.91 | 8173.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 8135.00 | 8170.93 | 8170.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 8157.50 | 8170.93 | 8170.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 8204.50 | 8177.64 | 8173.25 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 12:15:00 | 8136.50 | 8164.41 | 8167.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 14:15:00 | 8120.50 | 8154.92 | 8162.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 15:15:00 | 8128.50 | 8125.44 | 8140.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 15:15:00 | 8128.50 | 8125.44 | 8140.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 8128.50 | 8125.44 | 8140.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 8079.50 | 8125.44 | 8140.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 8123.50 | 8125.05 | 8139.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 8140.00 | 8125.05 | 8139.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 8032.50 | 8106.54 | 8129.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 7917.50 | 8079.56 | 8106.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 8007.50 | 7997.42 | 8000.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 7976.50 | 7865.09 | 7877.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 7938.00 | 7890.06 | 7887.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 7938.00 | 7890.06 | 7887.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 8026.50 | 7931.65 | 7908.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 8687.00 | 8721.63 | 8611.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 8687.00 | 8721.63 | 8611.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 8772.00 | 8729.92 | 8634.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 8900.00 | 8773.99 | 8672.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 8875.00 | 8786.79 | 8687.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 8876.50 | 8802.73 | 8703.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:00:00 | 8885.50 | 8785.20 | 8758.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 9375.00 | 8927.62 | 8837.57 | EMA400 retest candle locked (from upside) |
| Target hit | 2025-05-26 14:15:00 | 9790.00 | 8927.62 | 8837.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 9783.00 | 9885.55 | 9892.64 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 10085.50 | 9904.74 | 9885.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 10150.50 | 10029.16 | 9957.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 10084.00 | 10115.78 | 10027.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 10084.00 | 10115.78 | 10027.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 10257.00 | 10241.34 | 10172.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 10315.00 | 10254.37 | 10184.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 10284.50 | 10268.10 | 10203.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 10284.50 | 10273.18 | 10217.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 10288.00 | 10273.18 | 10217.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 10152.50 | 10252.54 | 10217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 10098.50 | 10252.54 | 10217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 10190.50 | 10240.13 | 10215.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 10164.00 | 10240.13 | 10215.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 15:15:00 | 10119.50 | 10178.25 | 10190.80 | Break + close below crossover candle low |

### Cycle 87 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 10374.50 | 10217.50 | 10207.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 10448.50 | 10296.10 | 10252.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 10329.50 | 10330.86 | 10281.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:00:00 | 10329.50 | 10330.86 | 10281.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 10331.00 | 10330.89 | 10286.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 10324.50 | 10330.89 | 10286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 10360.00 | 10336.71 | 10293.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 10313.50 | 10336.71 | 10293.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 10419.00 | 10419.79 | 10367.96 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 10:15:00 | 10196.50 | 10339.86 | 10345.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 10123.00 | 10271.23 | 10311.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 10:15:00 | 10119.00 | 10113.32 | 10205.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:45:00 | 10112.00 | 10113.32 | 10205.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 10107.50 | 10078.32 | 10144.54 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 10315.00 | 10188.71 | 10180.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 10345.50 | 10220.07 | 10195.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 10861.50 | 10863.51 | 10749.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 10861.50 | 10863.51 | 10749.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 10879.00 | 10875.77 | 10784.62 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 10550.00 | 10744.07 | 10756.44 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 10790.00 | 10733.19 | 10731.06 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 10697.00 | 10736.87 | 10738.50 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 10845.00 | 10750.28 | 10743.34 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 10690.00 | 10745.70 | 10748.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 10662.00 | 10721.33 | 10736.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 10705.00 | 10703.38 | 10723.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 10705.00 | 10703.38 | 10723.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 10796.00 | 10718.96 | 10726.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 10796.00 | 10718.96 | 10726.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 10739.00 | 10722.97 | 10727.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 10700.00 | 10716.78 | 10724.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 10825.00 | 10579.18 | 10615.90 | SL hit (close>static) qty=1.00 sl=10796.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 10768.00 | 10655.15 | 10646.43 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 10583.00 | 10672.63 | 10672.73 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 10725.00 | 10681.09 | 10676.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 11033.00 | 10751.47 | 10708.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 11179.00 | 10978.90 | 10912.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 10935.00 | 11106.56 | 11112.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 10935.00 | 11106.56 | 11112.63 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 11250.00 | 11111.18 | 11105.94 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 11050.00 | 11130.38 | 11131.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 10991.00 | 11092.04 | 11112.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 10855.00 | 10791.85 | 10901.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 10855.00 | 10791.85 | 10901.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 10532.00 | 10500.77 | 10604.03 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 10989.00 | 10665.21 | 10630.51 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 10480.00 | 10666.22 | 10681.26 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 10680.00 | 10644.25 | 10643.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 10797.00 | 10696.01 | 10668.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 10691.00 | 10707.00 | 10679.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 10612.00 | 10688.00 | 10673.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 10612.00 | 10688.00 | 10673.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 10633.00 | 10677.00 | 10669.54 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 10604.00 | 10652.96 | 10659.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 10538.00 | 10612.62 | 10637.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 10650.00 | 10611.68 | 10632.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 10650.00 | 10619.34 | 10633.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 10551.00 | 10619.34 | 10633.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 10371.00 | 10345.21 | 10413.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:15:00 | 10424.00 | 10345.21 | 10413.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 10422.00 | 10360.57 | 10413.99 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 10509.00 | 10438.98 | 10433.01 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 10298.00 | 10419.30 | 10431.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 10279.00 | 10358.38 | 10395.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:15:00 | 10447.00 | 10344.16 | 10382.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 10516.00 | 10378.53 | 10394.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 10516.00 | 10378.53 | 10394.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 10513.00 | 10405.42 | 10404.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 10546.00 | 10433.54 | 10417.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 10378.00 | 10453.76 | 10435.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 10300.00 | 10423.01 | 10423.58 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 10445.00 | 10419.42 | 10416.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 10506.00 | 10436.74 | 10424.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 10513.00 | 10521.48 | 10486.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 10513.00 | 10521.48 | 10486.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 10460.00 | 10509.18 | 10484.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 10460.00 | 10509.18 | 10484.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 10425.00 | 10492.35 | 10478.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 10400.00 | 10492.35 | 10478.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 10400.00 | 10462.62 | 10467.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 10371.00 | 10437.48 | 10454.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 10417.00 | 10415.32 | 10438.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:45:00 | 10410.00 | 10415.32 | 10438.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 10399.00 | 10398.65 | 10424.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 10315.00 | 10375.33 | 10402.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 10112.00 | 10107.04 | 10106.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 10112.00 | 10107.04 | 10106.92 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 10099.00 | 10105.43 | 10106.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 13:15:00 | 10070.00 | 10098.35 | 10102.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 10209.00 | 10122.38 | 10111.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 10552.00 | 10225.59 | 10178.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 10400.00 | 10471.65 | 10361.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 10382.00 | 10439.70 | 10365.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 10382.00 | 10439.70 | 10365.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 10425.00 | 10439.70 | 10365.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 10371.00 | 10425.96 | 10365.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 10360.00 | 10425.96 | 10365.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 10460.00 | 10432.77 | 10374.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 10383.00 | 10432.77 | 10374.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 10384.00 | 10427.98 | 10387.06 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 10305.00 | 10358.89 | 10366.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 10255.00 | 10338.11 | 10355.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 10363.00 | 10290.39 | 10317.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 10281.00 | 10288.51 | 10314.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 10238.00 | 10272.41 | 10304.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 10208.00 | 10272.41 | 10304.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 10384.00 | 10064.51 | 10023.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 10384.00 | 10064.51 | 10023.50 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 9993.00 | 10071.60 | 10078.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 9897.00 | 9984.54 | 10027.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 9713.00 | 9612.96 | 9709.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 9610.00 | 9612.37 | 9700.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 9542.00 | 9597.10 | 9685.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:15:00 | 9575.00 | 9592.14 | 9667.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 9897.00 | 9597.85 | 9588.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 9897.00 | 9597.85 | 9588.13 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 9463.50 | 9579.30 | 9594.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 9422.00 | 9501.44 | 9546.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 9479.00 | 9430.97 | 9469.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 9493.00 | 9443.38 | 9471.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 9437.50 | 9443.38 | 9471.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 9504.00 | 9441.89 | 9453.70 | SL hit (close>static) qty=1.00 sl=9493.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 9548.00 | 9463.11 | 9462.27 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 9422.50 | 9479.38 | 9484.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 9405.00 | 9464.50 | 9477.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 9505.00 | 9465.60 | 9475.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 9498.50 | 9472.18 | 9477.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 9496.00 | 9472.18 | 9477.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 9430.50 | 9463.85 | 9473.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 9372.00 | 9454.48 | 9468.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 9484.50 | 9378.20 | 9368.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 9484.50 | 9378.20 | 9368.53 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 9350.00 | 9368.10 | 9370.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 9303.00 | 9347.42 | 9357.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 15:15:00 | 9191.50 | 9184.05 | 9228.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 9158.50 | 9184.05 | 9228.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 9105.00 | 9088.02 | 9124.20 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 9239.50 | 9138.96 | 9136.31 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 9009.00 | 9113.64 | 9125.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 8995.00 | 9089.91 | 9113.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 8497.50 | 8457.65 | 8456.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 8533.50 | 8472.82 | 8463.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 8400.00 | 8522.77 | 8508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 8394.00 | 8497.01 | 8497.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 8376.50 | 8472.91 | 8486.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 15:15:00 | 8371.00 | 8350.86 | 8390.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 8413.00 | 8350.86 | 8390.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 8410.50 | 8362.79 | 8392.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 8340.50 | 8398.48 | 8402.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 8415.50 | 8370.53 | 8365.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 8415.50 | 8370.53 | 8365.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 8464.00 | 8400.74 | 8380.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 8365.50 | 8388.59 | 8376.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 8352.50 | 8381.37 | 8374.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 8336.00 | 8381.37 | 8374.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 12:15:00 | 8331.50 | 8367.02 | 8369.05 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 8401.50 | 8373.91 | 8372.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 8418.50 | 8396.59 | 8384.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 8647.00 | 8680.19 | 8586.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:30:00 | 8658.50 | 8680.19 | 8586.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 8639.50 | 8656.05 | 8603.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 8682.00 | 8656.05 | 8603.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 8655.50 | 8671.43 | 8629.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 8575.00 | 8649.69 | 8632.54 | SL hit (close<static) qty=1.00 sl=8600.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 8491.00 | 8601.84 | 8612.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 8386.00 | 8479.96 | 8541.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 8360.00 | 8327.37 | 8407.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 8350.00 | 8326.59 | 8400.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 8380.00 | 8341.65 | 8389.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 8385.00 | 8341.65 | 8389.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 8361.50 | 8345.62 | 8386.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 8352.50 | 8345.62 | 8386.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 8547.50 | 8385.99 | 8401.22 | SL hit (close>static) qty=1.00 sl=8392.50 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 8308.00 | 8280.51 | 8277.75 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 8192.50 | 8265.26 | 8271.91 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 8298.00 | 8267.77 | 8265.77 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 8239.00 | 8262.02 | 8263.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 8195.50 | 8248.71 | 8257.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 14:15:00 | 8115.00 | 8102.90 | 8151.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:30:00 | 8138.00 | 8102.90 | 8151.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 8150.00 | 8112.32 | 8151.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 8152.00 | 8112.32 | 8151.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 8170.00 | 8123.86 | 8153.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 8170.00 | 8123.86 | 8153.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 8119.50 | 8122.98 | 8150.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:30:00 | 8100.00 | 8112.51 | 8140.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 8025.00 | 8003.73 | 8001.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 8025.00 | 8003.73 | 8001.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 8127.50 | 8028.49 | 8012.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 8396.00 | 8405.98 | 8306.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 8396.00 | 8405.98 | 8306.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 8345.00 | 8390.94 | 8324.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 8345.00 | 8390.94 | 8324.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 8331.50 | 8379.05 | 8325.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 8324.50 | 8379.05 | 8325.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 8356.50 | 8374.54 | 8327.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 8271.00 | 8374.54 | 8327.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 8295.00 | 8358.63 | 8324.89 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 8235.50 | 8304.80 | 8305.93 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 8338.00 | 8311.44 | 8308.85 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 8184.00 | 8285.95 | 8297.50 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 8357.00 | 8301.86 | 8298.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 8380.00 | 8317.49 | 8305.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 8310.00 | 8318.95 | 8308.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 8293.00 | 8313.76 | 8307.01 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 8258.00 | 8293.69 | 8298.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 8208.50 | 8276.65 | 8290.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 8256.50 | 8238.88 | 8261.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 8220.00 | 8235.11 | 8258.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 8215.00 | 8235.11 | 8258.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 8174.50 | 8218.34 | 8242.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 8140.00 | 8193.80 | 8217.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 8144.00 | 8183.84 | 8210.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 8094.50 | 8149.38 | 8186.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:30:00 | 8139.00 | 8123.31 | 8156.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 8050.00 | 8100.73 | 8130.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 8030.00 | 8083.69 | 8120.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 7930.00 | 8019.03 | 8023.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 7905.00 | 7996.23 | 8012.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 7969.00 | 7939.66 | 7971.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 7921.00 | 7935.93 | 7967.15 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 8124.00 | 7993.95 | 7989.37 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 7979.00 | 8005.20 | 8005.81 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 12:15:00 | 8019.00 | 8007.96 | 8007.01 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 7955.00 | 7998.42 | 8002.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 7916.00 | 7981.93 | 7995.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 7917.00 | 7911.52 | 7944.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:15:00 | 7855.00 | 7911.52 | 7944.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 7846.50 | 7875.04 | 7917.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 7846.50 | 7875.04 | 7917.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 7794.00 | 7851.48 | 7888.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 8223.00 | 7908.69 | 7900.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 147 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 8223.00 | 7908.69 | 7900.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 8610.00 | 8157.65 | 8026.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 8668.50 | 8676.20 | 8511.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 8746.50 | 8676.20 | 8511.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 8735.00 | 8771.80 | 8693.94 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 8610.50 | 8667.24 | 8672.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 8590.00 | 8642.90 | 8658.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:15:00 | 8713.50 | 8637.77 | 8653.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 8674.00 | 8645.02 | 8654.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 8732.50 | 8645.02 | 8654.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 8712.00 | 8658.42 | 8660.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 8701.00 | 8658.42 | 8660.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 8738.50 | 8674.43 | 8667.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 8775.00 | 8702.04 | 8682.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 8770.00 | 8730.30 | 8708.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 8779.00 | 8736.30 | 8716.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 8757.00 | 8767.05 | 8753.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 8679.00 | 8716.60 | 8730.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 8573.00 | 8526.07 | 8577.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 8573.00 | 8526.07 | 8577.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 8579.50 | 8536.76 | 8577.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 8574.50 | 8536.76 | 8577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 8674.00 | 8564.20 | 8586.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 8674.00 | 8564.20 | 8586.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 8628.50 | 8577.06 | 8590.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:30:00 | 8664.00 | 8577.06 | 8590.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 8629.00 | 8601.52 | 8599.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 8675.00 | 8616.22 | 8606.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 8596.00 | 8612.17 | 8605.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 8623.50 | 8614.44 | 8607.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 8650.00 | 8619.85 | 8611.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 8558.50 | 8604.90 | 8607.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 8558.50 | 8604.90 | 8607.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 8534.00 | 8590.72 | 8601.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 8513.50 | 8475.11 | 8514.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 8565.50 | 8493.18 | 8519.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 8565.50 | 8493.18 | 8519.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 8578.00 | 8510.15 | 8524.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:30:00 | 8616.50 | 8510.15 | 8524.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 8667.00 | 8541.52 | 8537.69 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 8443.00 | 8531.97 | 8540.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 8438.00 | 8502.22 | 8524.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 8474.50 | 8471.80 | 8497.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 8474.50 | 8471.80 | 8497.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 8437.50 | 8437.42 | 8468.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 8402.00 | 8431.33 | 8462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 8404.00 | 8424.57 | 8456.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 8354.50 | 8406.50 | 8439.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 8360.00 | 8408.60 | 8437.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 8427.00 | 8413.71 | 8435.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 8409.00 | 8413.71 | 8435.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 8416.00 | 8414.16 | 8433.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 8395.00 | 8410.53 | 8428.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7981.90 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7983.80 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7975.25 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 8069.00 | 8054.71 | 8152.02 | SL hit (close>ema200) qty=0.50 sl=8054.71 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 8189.00 | 8074.79 | 8072.60 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 7997.00 | 8081.43 | 8086.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 7985.00 | 8050.24 | 8070.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 8001.00 | 7986.29 | 8022.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:45:00 | 7994.50 | 7986.29 | 8022.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 8012.50 | 7969.28 | 7998.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:45:00 | 8018.50 | 7969.28 | 7998.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 8020.00 | 7979.42 | 8000.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 7934.50 | 7960.94 | 7990.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 14:45:00 | 7894.50 | 7947.95 | 7981.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 7917.50 | 7961.36 | 7984.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 8118.00 | 7997.80 | 7979.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 7940.00 | 8014.62 | 7996.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 7900.00 | 7991.70 | 7987.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 7893.50 | 7991.70 | 7987.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 7916.00 | 7976.56 | 7981.44 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 8134.00 | 7997.22 | 7980.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 8147.50 | 8048.20 | 8007.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 8004.00 | 8039.57 | 8011.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 7968.00 | 8025.26 | 8007.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 7830.00 | 8025.26 | 8007.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 7781.00 | 7976.40 | 7986.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 7751.00 | 7841.75 | 7903.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 7817.00 | 7810.80 | 7867.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 7813.00 | 7810.80 | 7867.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 7835.00 | 7792.18 | 7837.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 7736.50 | 7801.29 | 7828.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 7349.67 | 7508.13 | 7629.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 7464.50 | 7376.12 | 7486.18 | SL hit (close>ema200) qty=0.50 sl=7376.12 alert=retest2 |

### Cycle 161 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 7580.00 | 7487.68 | 7483.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 7686.00 | 7565.12 | 7535.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 7620.00 | 7648.47 | 7603.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 7602.50 | 7639.45 | 7614.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 7650.00 | 7629.69 | 7613.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 8061.00 | 8138.83 | 8138.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 8061.00 | 8138.83 | 8138.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 7982.00 | 8084.99 | 8112.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 8085.50 | 8018.50 | 8056.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 8079.00 | 8030.60 | 8058.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 8018.00 | 8028.08 | 8054.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 8019.00 | 7949.82 | 7947.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 8019.00 | 7949.82 | 7947.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 8065.50 | 7972.96 | 7958.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 10:45:00 | 6740.15 | 2024-05-16 15:15:00 | 6830.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-23 13:15:00 | 6931.85 | 2024-05-28 13:15:00 | 6884.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-05-24 14:45:00 | 6932.90 | 2024-05-28 13:15:00 | 6884.90 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-05-27 09:15:00 | 7088.00 | 2024-05-28 13:15:00 | 6884.90 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-06-03 11:45:00 | 7219.00 | 2024-06-04 10:15:00 | 6950.40 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-06-04 09:45:00 | 7184.95 | 2024-06-04 10:15:00 | 6950.40 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-06-13 10:00:00 | 7899.50 | 2024-06-14 12:15:00 | 7729.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-06-13 11:30:00 | 7862.50 | 2024-06-14 12:15:00 | 7729.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-06-14 09:45:00 | 7867.65 | 2024-06-14 12:15:00 | 7729.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-06-21 14:30:00 | 7501.40 | 2024-06-24 09:15:00 | 7595.75 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-06-24 13:00:00 | 7483.65 | 2024-07-08 12:15:00 | 7354.60 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-06-25 12:15:00 | 7505.00 | 2024-07-08 12:15:00 | 7354.60 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2024-06-25 13:30:00 | 7499.95 | 2024-07-08 12:15:00 | 7354.60 | STOP_HIT | 1.00 | 1.94% |
| SELL | retest2 | 2024-06-27 12:30:00 | 7420.00 | 2024-07-08 12:15:00 | 7354.60 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-07-01 10:45:00 | 7368.80 | 2024-07-08 12:15:00 | 7354.60 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-07-19 13:15:00 | 7799.95 | 2024-07-19 14:15:00 | 7695.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-07-23 09:45:00 | 7630.70 | 2024-07-23 10:15:00 | 7740.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-07-25 14:15:00 | 7789.50 | 2024-07-31 15:15:00 | 7815.15 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-08-20 13:45:00 | 8001.00 | 2024-08-26 09:15:00 | 8801.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 15:00:00 | 8049.95 | 2024-08-26 09:15:00 | 8854.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-17 09:30:00 | 8832.45 | 2024-09-18 09:15:00 | 9011.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-09-20 09:15:00 | 9073.25 | 2024-09-20 13:15:00 | 8880.05 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-03 14:45:00 | 8430.05 | 2024-10-07 10:15:00 | 8567.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-10-04 09:15:00 | 8365.45 | 2024-10-07 10:15:00 | 8567.40 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-10-04 11:00:00 | 8434.00 | 2024-10-07 10:15:00 | 8567.40 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-10-04 14:30:00 | 8431.65 | 2024-10-07 10:15:00 | 8567.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-10-16 09:30:00 | 9138.75 | 2024-10-16 11:15:00 | 8982.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-21 12:00:00 | 8723.85 | 2024-10-22 14:15:00 | 8287.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 8687.00 | 2024-10-23 09:15:00 | 8252.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 8723.85 | 2024-10-23 11:15:00 | 8478.90 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-10-21 12:30:00 | 8687.00 | 2024-10-23 11:15:00 | 8478.90 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-11-14 14:00:00 | 9583.05 | 2024-11-25 09:15:00 | 9628.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-11-27 10:45:00 | 10120.00 | 2024-11-28 09:15:00 | 9751.60 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-12-02 15:00:00 | 9740.00 | 2024-12-04 10:15:00 | 9856.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-12-03 12:00:00 | 9775.00 | 2024-12-04 10:15:00 | 9856.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-03 13:15:00 | 9777.45 | 2024-12-04 10:15:00 | 9856.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-03 13:45:00 | 9777.85 | 2024-12-04 10:15:00 | 9856.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-12-17 12:30:00 | 9679.25 | 2024-12-18 14:15:00 | 9784.90 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-15 09:30:00 | 9385.00 | 2025-01-21 14:15:00 | 9410.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-01-24 12:15:00 | 9537.15 | 2025-01-24 14:15:00 | 9389.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-01-24 13:00:00 | 9544.50 | 2025-01-24 14:15:00 | 9389.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-01-30 11:45:00 | 8650.55 | 2025-02-01 14:15:00 | 8880.15 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-01-30 13:15:00 | 8696.00 | 2025-02-01 14:15:00 | 8880.15 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-02-20 11:00:00 | 8611.25 | 2025-02-21 15:15:00 | 8201.00 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-02-20 12:00:00 | 8535.00 | 2025-02-21 15:15:00 | 8201.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-02-28 09:15:00 | 8053.75 | 2025-03-05 14:15:00 | 8018.25 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-03-10 12:30:00 | 7856.55 | 2025-03-11 14:15:00 | 7986.95 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-03-10 15:15:00 | 7860.00 | 2025-03-11 14:15:00 | 7986.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-03-13 11:45:00 | 7838.85 | 2025-03-17 09:15:00 | 8053.15 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-03-13 13:00:00 | 7833.80 | 2025-03-17 09:15:00 | 8053.15 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-03-20 09:15:00 | 8263.70 | 2025-03-21 14:15:00 | 8071.05 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-25 11:30:00 | 7934.05 | 2025-03-27 10:15:00 | 8100.70 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-03-26 09:30:00 | 7940.00 | 2025-03-27 10:15:00 | 8100.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-03-26 10:15:00 | 7965.70 | 2025-03-27 10:15:00 | 8100.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-27 10:00:00 | 7969.90 | 2025-03-27 10:15:00 | 8100.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-04-03 15:00:00 | 8079.50 | 2025-04-04 09:15:00 | 7952.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-04-15 09:15:00 | 8064.50 | 2025-04-16 11:15:00 | 7941.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-02 09:15:00 | 7917.50 | 2025-05-12 12:15:00 | 7938.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-06 11:45:00 | 8007.50 | 2025-05-12 12:15:00 | 7938.00 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-05-12 10:45:00 | 7976.50 | 2025-05-12 12:15:00 | 7938.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-05-21 11:45:00 | 8900.00 | 2025-05-26 14:15:00 | 9790.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 12:30:00 | 8875.00 | 2025-05-26 14:15:00 | 9762.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 14:15:00 | 8876.50 | 2025-05-26 14:15:00 | 9764.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 10:00:00 | 8885.50 | 2025-05-26 14:15:00 | 9774.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 13:15:00 | 9881.00 | 2025-06-09 09:15:00 | 9783.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-05 09:15:00 | 10062.00 | 2025-06-09 09:15:00 | 9783.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-06-13 10:30:00 | 10315.00 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-06-13 13:15:00 | 10284.50 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-13 14:30:00 | 10284.50 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-13 15:00:00 | 10288.00 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-09 11:30:00 | 10700.00 | 2025-07-11 09:15:00 | 10825.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-17 09:15:00 | 11179.00 | 2025-07-21 14:15:00 | 10935.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-25 15:00:00 | 10315.00 | 2025-09-01 11:15:00 | 10112.00 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2025-09-10 10:30:00 | 10238.00 | 2025-09-19 14:15:00 | 10384.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-10 11:00:00 | 10208.00 | 2025-09-19 14:15:00 | 10384.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-29 11:30:00 | 9542.00 | 2025-10-01 11:15:00 | 9897.00 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-29 14:15:00 | 9575.00 | 2025-10-01 11:15:00 | 9897.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-10-08 09:15:00 | 9437.50 | 2025-10-09 09:15:00 | 9504.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-15 09:15:00 | 9372.00 | 2025-10-17 10:15:00 | 9484.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-21 09:15:00 | 8340.50 | 2025-11-24 13:15:00 | 8415.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-28 14:15:00 | 8682.00 | 2025-12-01 14:15:00 | 8575.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-01 10:45:00 | 8655.50 | 2025-12-01 14:15:00 | 8575.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-04 14:15:00 | 8352.50 | 2025-12-04 14:15:00 | 8547.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-12-05 13:45:00 | 8352.00 | 2025-12-10 10:15:00 | 8308.00 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-12-10 09:15:00 | 8324.50 | 2025-12-10 10:15:00 | 8308.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-12-16 12:30:00 | 8100.00 | 2025-12-23 15:15:00 | 8025.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2026-01-06 09:15:00 | 8140.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2026-01-06 10:00:00 | 8144.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2026-01-06 13:00:00 | 8094.50 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-01-07 10:30:00 | 8139.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2026-01-08 10:30:00 | 8030.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest1 | 2026-01-28 09:15:00 | 7855.00 | 2026-01-29 13:15:00 | 8223.00 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-02-10 09:15:00 | 8770.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-10 12:45:00 | 8779.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-11 15:15:00 | 8757.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-18 15:15:00 | 8650.00 | 2026-02-19 11:15:00 | 8558.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-26 10:45:00 | 8402.00 | 2026-03-04 09:15:00 | 7981.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 8404.00 | 2026-03-04 09:15:00 | 7983.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:00:00 | 8354.50 | 2026-03-04 09:15:00 | 7975.25 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2026-02-26 10:45:00 | 8402.00 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2026-02-26 11:30:00 | 8404.00 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2026-02-26 15:00:00 | 8354.50 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2026-02-27 09:15:00 | 8360.00 | 2026-03-09 09:15:00 | 7936.77 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2026-02-27 13:30:00 | 8395.00 | 2026-03-09 09:15:00 | 7942.00 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2026-02-27 09:15:00 | 8360.00 | 2026-03-09 14:15:00 | 8038.50 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2026-02-27 13:30:00 | 8395.00 | 2026-03-09 14:15:00 | 8038.50 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2026-03-13 13:30:00 | 7934.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-03-13 14:45:00 | 7894.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-03-16 09:15:00 | 7917.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-03-25 14:15:00 | 7736.50 | 2026-03-30 09:15:00 | 7349.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 7736.50 | 2026-04-01 09:15:00 | 7464.50 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2026-04-10 09:15:00 | 7650.00 | 2026-04-29 13:15:00 | 8061.00 | STOP_HIT | 1.00 | 5.37% |
| SELL | retest2 | 2026-05-04 12:00:00 | 8018.00 | 2026-05-08 10:15:00 | 8019.00 | STOP_HIT | 1.00 | -0.01% |
