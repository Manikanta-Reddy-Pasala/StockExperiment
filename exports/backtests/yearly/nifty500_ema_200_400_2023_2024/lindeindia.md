# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 7765.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 11 |
| TARGET_HIT | 16 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 17
- **Target hits / Stop hits / Partials:** 16 / 19 / 11
- **Avg / median % per leg:** 4.10% / 5.00%
- **Sum % (uncompounded):** 188.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.49% | -5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 1 | 10 | 0 | -0.49% | -5.3% |
| SELL (all) | 35 | 28 | 80.0% | 15 | 9 | 11 | 5.55% | 194.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 28 | 80.0% | 15 | 9 | 11 | 5.55% | 194.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 29 | 63.0% | 16 | 19 | 11 | 4.10% | 188.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 11:15:00 | 5660.00 | 5871.88 | 5872.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 5585.90 | 5852.40 | 5862.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 10:15:00 | 5638.00 | 5634.18 | 5716.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 11:15:00 | 5710.00 | 5634.94 | 5716.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 5710.00 | 5634.94 | 5716.31 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-13 13:15:00 | 6194.65 | 5676.56 | 5675.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 10:15:00 | 6372.75 | 5694.63 | 5684.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 8051.60 | 8302.13 | 7624.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 8051.60 | 8302.13 | 7624.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 8408.50 | 8413.64 | 7782.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 8272.20 | 8413.64 | 7782.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 7951.40 | 8405.76 | 7790.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:45:00 | 7913.70 | 8405.76 | 7790.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 7962.65 | 8401.35 | 7791.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:30:00 | 7831.85 | 8401.35 | 7791.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 8169.00 | 8672.54 | 8165.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 8169.00 | 8672.54 | 8165.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 8238.00 | 8668.22 | 8165.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 8220.05 | 8668.22 | 8165.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 8196.80 | 8632.35 | 8174.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:30:00 | 8176.05 | 8632.35 | 8174.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 8181.10 | 8627.86 | 8174.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 11:30:00 | 8273.65 | 8593.99 | 8174.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 13:00:00 | 8230.55 | 8590.38 | 8174.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 8160.25 | 8582.04 | 8174.79 | SL hit (close<static) qty=1.00 sl=8167.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 7716.30 | 8229.40 | 8230.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 7652.00 | 8213.53 | 8222.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 7579.95 | 7526.26 | 7754.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 7579.95 | 7526.26 | 7754.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 7759.00 | 7530.12 | 7752.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 7759.00 | 7530.12 | 7752.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 7752.70 | 7532.34 | 7752.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:30:00 | 7800.05 | 7532.34 | 7752.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 7905.00 | 7536.04 | 7753.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 7913.45 | 7536.04 | 7753.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 7841.10 | 7539.08 | 7754.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:30:00 | 7855.55 | 7539.08 | 7754.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 8119.30 | 7561.79 | 7749.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 8119.30 | 7561.79 | 7749.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 8118.80 | 7567.33 | 7751.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:45:00 | 8119.00 | 7567.33 | 7751.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 12:15:00 | 8658.70 | 7892.12 | 7888.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 10:15:00 | 8732.50 | 7930.27 | 7908.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 8067.00 | 8080.34 | 7995.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 14:00:00 | 8067.00 | 8080.34 | 7995.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 7999.95 | 8079.54 | 7995.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 7999.95 | 8079.54 | 7995.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 8030.00 | 8079.05 | 7995.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 8178.00 | 8079.05 | 7995.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 7900.05 | 8078.36 | 7996.06 | SL hit (close<static) qty=1.00 sl=7975.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 7676.00 | 7999.51 | 8000.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 7518.00 | 7914.28 | 7952.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 14:15:00 | 6607.95 | 6537.32 | 6903.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-06 15:00:00 | 6607.95 | 6537.32 | 6903.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6265.00 | 6137.62 | 6503.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 6537.25 | 6137.62 | 6503.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 6356.60 | 5989.97 | 6264.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:00:00 | 6356.60 | 5989.97 | 6264.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 6389.85 | 5993.95 | 6265.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:30:00 | 6397.20 | 5993.95 | 6265.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 6308.90 | 5999.99 | 6265.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:45:00 | 6347.00 | 5999.99 | 6265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 6165.00 | 6004.31 | 6265.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:45:00 | 6314.85 | 6004.31 | 6265.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 6175.00 | 6007.14 | 6264.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 6223.35 | 6007.14 | 6264.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 6243.35 | 6022.06 | 6259.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 6263.35 | 6022.06 | 6259.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 6300.35 | 6024.83 | 6259.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 6300.35 | 6024.83 | 6259.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 6347.80 | 6028.04 | 6259.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 6347.80 | 6028.04 | 6259.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 6419.80 | 6037.44 | 6261.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 6419.80 | 6037.44 | 6261.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 6389.65 | 6040.95 | 6261.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 6285.00 | 6050.36 | 6263.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 5970.75 | 6055.26 | 6256.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 6078.80 | 6033.52 | 6231.55 | SL hit (close>ema200) qty=0.50 sl=6033.52 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 6370.00 | 6219.15 | 6219.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 6458.00 | 6229.37 | 6224.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 6199.00 | 6272.67 | 6248.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 6199.00 | 6272.67 | 6248.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 6199.00 | 6272.67 | 6248.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 6195.00 | 6272.67 | 6248.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 6170.00 | 6271.65 | 6248.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:45:00 | 6167.00 | 6270.01 | 6247.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 6222.50 | 6268.27 | 6247.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 6222.50 | 6268.27 | 6247.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 6185.00 | 6264.64 | 6245.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 6178.00 | 6264.64 | 6245.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 6839.00 | 7129.09 | 6882.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 6839.00 | 7129.09 | 6882.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 6758.50 | 7125.40 | 6882.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 6758.50 | 7125.40 | 6882.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 6860.00 | 7056.80 | 6866.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 6754.50 | 7056.80 | 6866.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 6723.50 | 7053.48 | 6865.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 6670.50 | 7053.48 | 6865.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 6641.50 | 7049.38 | 6864.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 6641.50 | 7049.38 | 6864.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 6790.00 | 6839.46 | 6795.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 6898.50 | 6837.94 | 6795.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 6741.00 | 6834.93 | 6795.82 | SL hit (close<static) qty=1.00 sl=6755.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 6639.00 | 6773.76 | 6773.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 6606.00 | 6769.01 | 6771.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 6559.00 | 6466.98 | 6573.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 6559.00 | 6466.98 | 6573.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 6525.00 | 6467.56 | 6573.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 6490.50 | 6468.00 | 6573.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 6474.00 | 6467.35 | 6572.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 6480.00 | 6443.34 | 6530.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 6491.50 | 6446.73 | 6526.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 6528.00 | 6447.88 | 6521.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 6528.00 | 6447.88 | 6521.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 6513.50 | 6448.53 | 6521.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 6484.50 | 6455.83 | 6521.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 6491.00 | 6456.22 | 6521.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 6495.50 | 6457.00 | 6521.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 6487.00 | 6445.37 | 6506.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 6479.00 | 6446.25 | 6505.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 6400.00 | 6445.79 | 6505.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6165.97 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6150.30 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6156.00 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6166.92 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6160.27 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6166.45 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6170.72 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6162.65 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:15:00 | 6080.00 | 6253.85 | 6354.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-06 14:15:00 | 5841.45 | 6149.98 | 6269.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 6606.00 | 6004.07 | 6003.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 13:15:00 | 6909.00 | 6105.42 | 6056.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 10:15:00 | 6597.50 | 6769.09 | 6532.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 11:00:00 | 6597.50 | 6769.09 | 6532.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-27 11:30:00 | 8273.65 | 2024-06-27 14:15:00 | 8160.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-06-27 13:00:00 | 8230.55 | 2024-06-27 14:15:00 | 8160.25 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-06-27 14:45:00 | 8225.05 | 2024-06-27 15:15:00 | 8161.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-28 09:15:00 | 8300.10 | 2024-07-04 12:15:00 | 9130.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-18 14:00:00 | 8409.60 | 2024-07-18 15:15:00 | 8345.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-07-18 14:45:00 | 8455.75 | 2024-07-18 15:15:00 | 8345.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-07-31 10:00:00 | 8440.60 | 2024-07-31 10:15:00 | 8326.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-07 09:15:00 | 8178.00 | 2024-10-07 10:15:00 | 7900.05 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-10-08 14:45:00 | 8078.25 | 2024-10-22 14:15:00 | 7950.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-22 10:45:00 | 8084.50 | 2024-10-22 14:15:00 | 7950.80 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-25 15:15:00 | 6285.00 | 2025-02-28 09:15:00 | 5970.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 15:15:00 | 6285.00 | 2025-03-04 09:15:00 | 6078.80 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-03-20 12:45:00 | 6303.50 | 2025-03-21 11:15:00 | 6450.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-20 15:15:00 | 6296.00 | 2025-03-21 11:15:00 | 6450.10 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-03-25 10:00:00 | 6316.05 | 2025-03-28 09:15:00 | 6231.50 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-03-26 14:45:00 | 6165.00 | 2025-03-28 09:15:00 | 6231.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-03-27 10:15:00 | 6167.00 | 2025-03-28 09:15:00 | 6231.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-27 15:00:00 | 6161.20 | 2025-03-28 13:15:00 | 6343.70 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-03-28 13:15:00 | 6174.75 | 2025-04-04 10:15:00 | 6000.25 | PARTIAL | 0.50 | 2.83% |
| SELL | retest2 | 2025-03-28 13:15:00 | 6174.75 | 2025-04-04 14:15:00 | 5684.45 | TARGET_HIT | 0.50 | 7.94% |
| SELL | retest2 | 2025-04-01 10:15:00 | 6196.20 | 2025-04-04 14:15:00 | 5576.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-01 15:00:00 | 6210.40 | 2025-04-04 14:15:00 | 5589.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 14:45:00 | 6207.90 | 2025-04-04 14:15:00 | 5587.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 11:30:00 | 6216.00 | 2025-04-04 14:15:00 | 5594.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 13:30:00 | 6190.00 | 2025-04-04 14:15:00 | 5571.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-16 11:30:00 | 6194.00 | 2025-04-16 14:15:00 | 6242.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-04-16 15:15:00 | 6202.00 | 2025-04-17 09:15:00 | 6243.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-08 15:00:00 | 6898.50 | 2025-07-10 09:15:00 | 6741.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-08-26 13:45:00 | 6490.50 | 2025-10-06 10:15:00 | 6165.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 14:45:00 | 6474.00 | 2025-10-06 10:15:00 | 6150.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 6480.00 | 2025-10-06 10:15:00 | 6156.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 11:00:00 | 6491.50 | 2025-10-06 10:15:00 | 6166.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-10-06 10:15:00 | 6160.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-10-06 10:15:00 | 6166.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-10-06 10:15:00 | 6170.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-10-06 10:15:00 | 6162.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-10-24 09:15:00 | 6080.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 6490.50 | 2025-11-06 14:15:00 | 5841.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-26 14:45:00 | 6474.00 | 2025-11-06 14:15:00 | 5826.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 6480.00 | 2025-11-06 14:15:00 | 5832.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-11 11:00:00 | 6491.50 | 2025-11-06 14:15:00 | 5842.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-11-06 14:15:00 | 5836.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-11-06 14:15:00 | 5841.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-11-06 14:15:00 | 5845.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-11-06 14:15:00 | 5838.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-11-11 10:15:00 | 5760.00 | TARGET_HIT | 0.50 | 10.00% |
