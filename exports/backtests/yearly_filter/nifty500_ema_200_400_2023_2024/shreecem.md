# Shree Cement Ltd. (SHREECEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 25445.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 98 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 76 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 65
- **Target hits / Stop hits / Partials:** 2 / 78 / 10
- **Avg / median % per leg:** -0.53% / -1.37%
- **Sum % (uncompounded):** -47.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 5 | 16.1% | 2 | 29 | 0 | -0.90% | -27.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 5 | 16.1% | 2 | 29 | 0 | -0.90% | -27.9% |
| SELL (all) | 59 | 20 | 33.9% | 0 | 49 | 10 | -0.33% | -19.6% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.48% | -9.9% |
| SELL @ 3rd Alert (retest2) | 55 | 20 | 36.4% | 0 | 45 | 10 | -0.18% | -9.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.48% | -9.9% |
| retest2 (combined) | 86 | 25 | 29.1% | 2 | 74 | 10 | -0.44% | -37.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 15:15:00 | 24200.00 | 24834.54 | 24836.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 09:15:00 | 24155.00 | 24827.77 | 24832.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 13:15:00 | 24255.60 | 24198.37 | 24454.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 14:00:00 | 24255.60 | 24198.37 | 24454.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 24256.00 | 24155.22 | 24370.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 11:00:00 | 24135.90 | 24155.03 | 24369.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 13:15:00 | 24177.80 | 24155.59 | 24367.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 14:00:00 | 24175.00 | 24155.78 | 24366.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:30:00 | 24167.40 | 24160.24 | 24363.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 24350.10 | 24164.51 | 24355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 24350.10 | 24164.51 | 24355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 24189.20 | 24164.75 | 24354.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 09:15:00 | 24219.00 | 24164.75 | 24354.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 24139.50 | 24164.50 | 24353.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 10:15:00 | 24056.50 | 24164.50 | 24353.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 10:45:00 | 24066.10 | 24159.92 | 24344.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 11:30:00 | 24087.10 | 24159.36 | 24342.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 13:00:00 | 24080.00 | 24158.57 | 24341.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 23931.00 | 24009.49 | 24221.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 12:15:00 | 23785.00 | 24006.02 | 24217.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 10:00:00 | 23793.80 | 23992.63 | 24205.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 23765.90 | 23979.97 | 24192.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 13:00:00 | 23800.10 | 23969.86 | 24183.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 24085.00 | 23979.58 | 24170.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 15:15:00 | 24001.00 | 23984.61 | 24167.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 10:00:00 | 24022.10 | 23978.46 | 24156.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 24819.80 | 23988.04 | 24155.42 | SL hit (close>static) qty=1.00 sl=24400.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 11:15:00 | 25724.80 | 24314.38 | 24309.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 25800.30 | 24409.17 | 24357.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 11:15:00 | 25405.50 | 25471.89 | 25040.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 12:00:00 | 25405.50 | 25471.89 | 25040.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 12:15:00 | 25488.30 | 25956.49 | 25524.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 13:00:00 | 25488.30 | 25956.49 | 25524.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 13:15:00 | 25900.00 | 25955.93 | 25526.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 14:30:00 | 25943.90 | 25881.32 | 25531.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 15:00:00 | 25950.30 | 25881.32 | 25531.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 14:15:00 | 25431.60 | 25853.28 | 25540.72 | SL hit (close<static) qty=1.00 sl=25468.90 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 26527.65 | 27261.09 | 27264.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 26250.00 | 27191.92 | 27228.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 13:15:00 | 25681.90 | 25680.44 | 26216.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 14:00:00 | 25681.90 | 25680.44 | 26216.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 26179.45 | 25687.78 | 26209.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 10:30:00 | 26292.05 | 25687.78 | 26209.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 26198.30 | 25715.01 | 26188.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:00:00 | 26198.30 | 25715.01 | 26188.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 26424.50 | 25722.07 | 26189.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 12:00:00 | 26424.50 | 25722.07 | 26189.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 12:15:00 | 26429.85 | 25729.11 | 26190.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:00:00 | 26429.85 | 25729.11 | 26190.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 26532.20 | 25783.51 | 26198.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 15:00:00 | 26532.20 | 25783.51 | 26198.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 26528.00 | 25790.92 | 26199.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 26493.85 | 25790.92 | 26199.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 26145.00 | 25839.89 | 26207.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 12:00:00 | 26017.35 | 25841.65 | 26206.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 11:45:00 | 26023.95 | 25851.48 | 26186.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 09:15:00 | 26515.30 | 25863.03 | 26184.03 | SL hit (close>static) qty=1.00 sl=26245.35 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 09:15:00 | 27236.75 | 25677.50 | 25673.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 10:15:00 | 27476.70 | 25695.40 | 25682.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 27008.70 | 27022.31 | 26540.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 27008.70 | 27022.31 | 26540.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 27133.35 | 27421.29 | 26989.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:30:00 | 27148.55 | 27418.72 | 26990.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 26531.40 | 27440.08 | 27054.29 | SL hit (close<static) qty=1.00 sl=26960.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 24300.00 | 26747.19 | 26750.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 24226.00 | 26629.84 | 26690.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 25684.75 | 25544.72 | 25995.17 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 13:00:00 | 25458.15 | 25545.08 | 25988.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:30:00 | 25450.05 | 25539.77 | 25975.00 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:45:00 | 25461.40 | 25538.97 | 25972.43 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 13:00:00 | 25466.00 | 25538.24 | 25969.90 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 26089.10 | 25554.09 | 25954.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 26089.10 | 25554.09 | 25954.83 | SL hit (close>ema400) qty=1.00 sl=25954.83 alert=retest1 |

### Cycle 6 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 27371.80 | 25277.70 | 25271.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 27450.00 | 25318.75 | 25292.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 26582.80 | 26603.85 | 26096.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 09:30:00 | 26541.85 | 26603.85 | 26096.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 26054.90 | 26583.86 | 26106.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 26054.90 | 26583.86 | 26106.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 25956.95 | 26577.62 | 26105.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 25956.95 | 26577.62 | 26105.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 25951.25 | 26571.39 | 26104.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:45:00 | 25900.00 | 26571.39 | 26104.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 26069.00 | 26556.05 | 26104.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 25960.05 | 26556.05 | 26104.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 25987.00 | 26540.32 | 26102.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:30:00 | 25981.00 | 26540.32 | 26102.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 25922.80 | 26534.18 | 26101.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:30:00 | 25922.45 | 26534.18 | 26101.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 25800.00 | 26513.12 | 26097.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:00:00 | 26011.50 | 26385.27 | 26063.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 15:15:00 | 26000.00 | 26371.15 | 26084.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 12:30:00 | 26016.70 | 26357.06 | 26084.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 14:45:00 | 26060.15 | 26348.93 | 26082.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 26009.30 | 26345.55 | 26082.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 25942.75 | 26345.55 | 26082.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 26072.40 | 26340.72 | 26082.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 26072.40 | 26340.72 | 26082.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 26072.35 | 26338.05 | 26082.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 26104.65 | 26338.05 | 26082.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 26065.00 | 26335.34 | 26082.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 25247.00 | 26251.70 | 26060.86 | SL hit (close<static) qty=1.00 sl=25545.40 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 25237.90 | 25898.25 | 25901.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 25099.95 | 25872.53 | 25888.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 14:15:00 | 25905.95 | 25807.14 | 25853.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 14:15:00 | 25905.95 | 25807.14 | 25853.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 25905.95 | 25807.14 | 25853.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:45:00 | 26000.00 | 25807.14 | 25853.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 25800.00 | 25807.07 | 25853.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 26333.20 | 25807.07 | 25853.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 26126.45 | 25810.25 | 25854.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 25802.35 | 25816.37 | 25856.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 25769.75 | 25815.98 | 25856.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 12:45:00 | 25797.60 | 25777.76 | 25834.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 26399.05 | 25792.63 | 25840.52 | SL hit (close>static) qty=1.00 sl=26393.40 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 26790.75 | 25888.60 | 25887.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 27122.35 | 25927.66 | 25907.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 27273.15 | 27634.04 | 27039.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 10:00:00 | 27273.15 | 27634.04 | 27039.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 27278.00 | 27630.50 | 27040.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 27278.00 | 27630.50 | 27040.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 29140.00 | 29949.62 | 29133.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 29140.00 | 29949.62 | 29133.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 29085.00 | 29941.02 | 29132.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 29085.00 | 29941.02 | 29132.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 29265.00 | 29934.29 | 29133.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 15:00:00 | 29360.00 | 29928.58 | 29134.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 29450.00 | 29859.34 | 29163.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 28910.00 | 29811.42 | 29182.50 | SL hit (close<static) qty=1.00 sl=29060.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 29915.00 | 30442.66 | 30443.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 29710.00 | 30352.73 | 30396.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 30350.00 | 30333.99 | 30385.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 30350.00 | 30333.99 | 30385.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 30350.00 | 30333.99 | 30385.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 30350.00 | 30333.99 | 30385.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 30605.00 | 30336.68 | 30386.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 30605.00 | 30336.68 | 30386.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 30600.00 | 30339.30 | 30387.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:30:00 | 30585.00 | 30339.30 | 30387.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 30450.00 | 30343.35 | 30388.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 30450.00 | 30343.35 | 30388.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 30360.00 | 30343.51 | 30388.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 30270.00 | 30342.45 | 30387.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 30255.00 | 30336.94 | 30384.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 28756.50 | 29932.65 | 30123.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 28742.25 | 29932.65 | 30123.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 29705.00 | 29687.46 | 29942.54 | SL hit (close>ema200) qty=0.50 sl=29687.46 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-07 11:00:00 | 24135.90 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2023-08-07 13:15:00 | 24177.80 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2023-08-07 14:00:00 | 24175.00 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2023-08-08 11:30:00 | 24167.40 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2023-08-10 10:15:00 | 24056.50 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2023-08-11 10:45:00 | 24066.10 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2023-08-11 11:30:00 | 24087.10 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2023-08-11 13:00:00 | 24080.00 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2023-08-23 12:15:00 | 23785.00 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2023-08-24 10:00:00 | 23793.80 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2023-08-25 09:15:00 | 23765.90 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2023-08-25 13:00:00 | 23800.10 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2023-08-30 15:15:00 | 24001.00 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2023-09-01 10:00:00 | 24022.10 | 2023-09-04 09:15:00 | 24819.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2023-10-30 14:30:00 | 25943.90 | 2023-11-01 14:15:00 | 25431.60 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2023-10-30 15:00:00 | 25950.30 | 2023-11-01 14:15:00 | 25431.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-11-02 14:30:00 | 25939.70 | 2023-12-12 09:15:00 | 28533.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-03 09:30:00 | 25982.00 | 2023-12-12 09:15:00 | 28580.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-09 11:45:00 | 27628.50 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-02-09 14:00:00 | 27633.20 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-02-09 14:45:00 | 27602.20 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-02-12 09:15:00 | 27660.00 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-02-12 11:30:00 | 27350.10 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-02-12 12:15:00 | 27326.40 | 2024-02-13 12:15:00 | 27160.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-04-04 12:00:00 | 26017.35 | 2024-04-09 09:15:00 | 26515.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-04-08 11:45:00 | 26023.95 | 2024-04-09 09:15:00 | 26515.30 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-04-09 13:30:00 | 26048.60 | 2024-04-18 09:15:00 | 24746.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 14:30:00 | 26027.60 | 2024-04-18 09:15:00 | 24726.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 13:30:00 | 26048.60 | 2024-05-02 10:15:00 | 25190.95 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2024-04-09 14:30:00 | 26027.60 | 2024-05-02 10:15:00 | 25190.95 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-05-03 11:45:00 | 25487.15 | 2024-05-13 09:15:00 | 25756.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-05-06 11:15:00 | 25443.25 | 2024-05-13 09:15:00 | 25756.05 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-05-06 15:15:00 | 25458.35 | 2024-05-13 09:15:00 | 25756.05 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-05-07 10:15:00 | 25311.00 | 2024-05-13 09:15:00 | 25756.05 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-05-09 09:15:00 | 25534.30 | 2024-05-13 11:15:00 | 25946.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-05-09 10:15:00 | 25597.75 | 2024-05-13 11:15:00 | 25946.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-05-10 10:30:00 | 25596.95 | 2024-05-13 11:15:00 | 25946.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-10 11:30:00 | 25575.60 | 2024-05-13 11:15:00 | 25946.95 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-05-23 10:30:00 | 25486.30 | 2024-06-04 11:15:00 | 24211.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 12:45:00 | 25500.00 | 2024-06-04 11:15:00 | 24225.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 14:00:00 | 25500.00 | 2024-06-04 11:15:00 | 24225.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 25435.00 | 2024-06-04 11:15:00 | 24163.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 10:30:00 | 25486.30 | 2024-06-05 11:15:00 | 25419.85 | STOP_HIT | 0.50 | 0.26% |
| SELL | retest2 | 2024-05-23 12:45:00 | 25500.00 | 2024-06-05 11:15:00 | 25419.85 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2024-05-24 14:00:00 | 25500.00 | 2024-06-05 11:15:00 | 25419.85 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2024-05-29 09:15:00 | 25435.00 | 2024-06-05 11:15:00 | 25419.85 | STOP_HIT | 0.50 | 0.06% |
| BUY | retest2 | 2024-07-30 11:30:00 | 27148.55 | 2024-08-05 09:15:00 | 26531.40 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest1 | 2024-09-02 13:00:00 | 25458.15 | 2024-09-05 09:15:00 | 26089.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest1 | 2024-09-03 10:30:00 | 25450.05 | 2024-09-05 09:15:00 | 26089.10 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest1 | 2024-09-03 11:45:00 | 25461.40 | 2024-09-05 09:15:00 | 26089.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest1 | 2024-09-03 13:00:00 | 25466.00 | 2024-09-05 09:15:00 | 26089.10 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-09-05 11:15:00 | 25892.50 | 2024-09-25 09:15:00 | 26019.55 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-09-05 12:45:00 | 25903.00 | 2024-09-25 09:15:00 | 26019.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-09-10 13:15:00 | 25910.80 | 2024-09-26 11:15:00 | 25976.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-09-13 12:45:00 | 25912.55 | 2024-09-26 11:15:00 | 25976.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-09-16 11:00:00 | 25788.40 | 2024-09-26 14:15:00 | 26100.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-09-24 09:30:00 | 25787.95 | 2024-09-26 14:15:00 | 26100.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-09-25 13:45:00 | 25807.55 | 2024-09-26 14:15:00 | 26100.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-25 14:45:00 | 25810.45 | 2024-09-26 14:15:00 | 26100.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-10-07 10:30:00 | 25760.00 | 2024-10-11 10:15:00 | 24472.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 11:30:00 | 25705.75 | 2024-10-11 10:15:00 | 24420.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 10:30:00 | 25760.00 | 2024-10-24 09:15:00 | 25150.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-10-07 11:30:00 | 25705.75 | 2024-10-24 09:15:00 | 25150.00 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest2 | 2025-01-02 11:00:00 | 26011.50 | 2025-01-13 09:15:00 | 25247.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-01-06 15:15:00 | 26000.00 | 2025-01-13 09:15:00 | 25247.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-01-07 12:30:00 | 26016.70 | 2025-01-13 09:15:00 | 25247.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-01-07 14:45:00 | 26060.15 | 2025-01-13 09:15:00 | 25247.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-01-24 13:30:00 | 25802.35 | 2025-01-29 09:15:00 | 26399.05 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-01-24 14:30:00 | 25769.75 | 2025-01-29 09:15:00 | 26399.05 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-01-28 12:45:00 | 25797.60 | 2025-01-29 09:15:00 | 26399.05 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-05-02 15:00:00 | 29360.00 | 2025-05-09 09:15:00 | 28910.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-07 11:00:00 | 29450.00 | 2025-05-09 09:15:00 | 28910.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-09 11:30:00 | 29485.00 | 2025-06-11 15:15:00 | 29655.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-05-09 12:15:00 | 29405.00 | 2025-06-11 15:15:00 | 29655.00 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2025-05-12 09:15:00 | 29710.00 | 2025-06-11 15:15:00 | 29655.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-06-03 10:45:00 | 29545.00 | 2025-06-12 13:15:00 | 29670.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-06-03 12:00:00 | 29530.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-06-03 12:30:00 | 29610.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-06-10 09:30:00 | 29865.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-06-10 15:15:00 | 29985.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-11 14:00:00 | 29860.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-06-12 09:15:00 | 29915.00 | 2025-06-20 14:15:00 | 28980.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-26 10:00:00 | 29965.00 | 2025-08-28 14:15:00 | 29675.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-26 12:30:00 | 29970.00 | 2025-08-28 14:15:00 | 29675.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-26 10:00:00 | 29920.00 | 2025-08-28 14:15:00 | 29675.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-28 09:45:00 | 29905.00 | 2025-08-28 14:15:00 | 29675.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-09 13:30:00 | 30270.00 | 2025-09-29 09:15:00 | 28756.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 09:30:00 | 30255.00 | 2025-09-29 09:15:00 | 28742.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 13:30:00 | 30270.00 | 2025-10-09 10:15:00 | 29705.00 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2025-09-10 09:30:00 | 30255.00 | 2025-10-09 10:15:00 | 29705.00 | STOP_HIT | 0.50 | 1.82% |
