# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 5695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 5 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 23
- **Target hits / Stop hits / Partials:** 5 / 30 / 11
- **Avg / median % per leg:** 1.18% / 0.18%
- **Sum % (uncompounded):** 54.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.91% | 44.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.91% | 44.2% |
| SELL (all) | 37 | 18 | 48.6% | 0 | 26 | 11 | 0.27% | 9.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 18 | 48.6% | 0 | 26 | 11 | 0.27% | 9.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 23 | 50.0% | 5 | 30 | 11 | 1.18% | 54.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 6654.20 | 6196.17 | 6194.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 12:15:00 | 6704.00 | 6327.28 | 6266.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 09:15:00 | 6940.00 | 7115.20 | 6866.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 10:15:00 | 6882.40 | 7100.52 | 6868.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 6882.40 | 7100.52 | 6868.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:45:00 | 6872.10 | 7100.52 | 6868.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 6845.05 | 7097.98 | 6868.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:30:00 | 6850.00 | 7097.98 | 6868.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 6784.50 | 7094.86 | 6868.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:00:00 | 6784.50 | 7094.86 | 6868.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 10:15:00 | 6445.95 | 6717.91 | 6718.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 09:15:00 | 6424.10 | 6702.08 | 6710.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 11:15:00 | 6449.90 | 6438.26 | 6540.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-11 12:00:00 | 6449.90 | 6438.26 | 6540.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 6549.70 | 6437.85 | 6536.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:30:00 | 6530.00 | 6437.85 | 6536.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 13:15:00 | 6588.00 | 6439.34 | 6536.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 14:15:00 | 6684.00 | 6439.34 | 6536.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 6650.00 | 6603.56 | 6606.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 6650.00 | 6603.56 | 6606.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 6583.60 | 6604.39 | 6606.68 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 13:15:00 | 6678.85 | 6609.46 | 6609.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 6708.60 | 6612.81 | 6610.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 6625.60 | 6651.65 | 6632.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 12:15:00 | 6625.60 | 6651.65 | 6632.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 6625.60 | 6651.65 | 6632.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:00:00 | 6625.60 | 6651.65 | 6632.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 6666.90 | 6651.80 | 6632.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 12:30:00 | 6696.20 | 6652.06 | 6633.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 13:00:00 | 6709.65 | 6652.06 | 6633.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 15:15:00 | 6705.55 | 6658.41 | 6637.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:45:00 | 6757.25 | 6660.12 | 6638.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 6673.40 | 6673.29 | 6647.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 14:30:00 | 6643.20 | 6673.29 | 6647.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 6655.70 | 6680.40 | 6653.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:30:00 | 6630.45 | 6680.40 | 6653.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 6648.55 | 6680.08 | 6653.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:00:00 | 6648.55 | 6680.08 | 6653.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 6700.00 | 6680.28 | 6653.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:30:00 | 6691.10 | 6680.28 | 6653.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 6652.00 | 6684.39 | 6656.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:45:00 | 6651.50 | 6683.86 | 6656.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 6628.10 | 6683.30 | 6656.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:00:00 | 6628.10 | 6683.30 | 6656.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-23 12:15:00 | 6620.00 | 6682.17 | 6656.30 | SL hit (close<static) qty=1.00 sl=6625.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 11:15:00 | 6337.70 | 6632.79 | 6633.62 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 6758.60 | 6626.72 | 6626.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 6823.60 | 6629.92 | 6627.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 11:15:00 | 6686.00 | 6689.06 | 6660.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 12:00:00 | 6686.00 | 6689.06 | 6660.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 6682.00 | 6688.99 | 6660.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:30:00 | 6715.00 | 6689.40 | 6661.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 12:15:00 | 6719.10 | 6689.63 | 6662.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 13:45:00 | 6705.00 | 6689.62 | 6662.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:15:00 | 6710.05 | 6689.60 | 6662.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-08 09:15:00 | 7386.50 | 6898.05 | 6787.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 10:15:00 | 6468.05 | 7042.45 | 7042.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 14:15:00 | 6390.00 | 6954.28 | 6996.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 14:15:00 | 5965.15 | 5958.49 | 6240.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 15:00:00 | 5965.15 | 5958.49 | 6240.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 6173.00 | 5972.03 | 6225.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 10:00:00 | 6126.85 | 5983.95 | 6223.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 14:00:00 | 6123.65 | 5990.47 | 6221.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 09:30:00 | 6125.00 | 5999.05 | 6214.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:45:00 | 6116.10 | 6014.13 | 6166.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 6193.00 | 6017.20 | 6166.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:45:00 | 6228.10 | 6017.20 | 6166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 6229.75 | 6019.32 | 6166.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 10:45:00 | 6288.65 | 6019.32 | 6166.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 6284.85 | 6024.17 | 6167.81 | SL hit (close>static) qty=1.00 sl=6249.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 7178.05 | 6256.53 | 6254.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 7283.15 | 6319.84 | 6286.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 12:15:00 | 6948.15 | 6981.80 | 6733.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 6948.15 | 6981.80 | 6733.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 6948.15 | 6981.80 | 6733.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 12:30:00 | 7093.60 | 6982.58 | 6742.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-10 14:15:00 | 7802.96 | 7067.71 | 6813.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 7979.75 | 8147.97 | 8148.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 7891.30 | 8134.17 | 8141.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 7698.00 | 7665.16 | 7828.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 14:45:00 | 7736.15 | 7665.16 | 7828.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 7737.50 | 7666.74 | 7826.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:45:00 | 7727.50 | 7668.79 | 7825.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:30:00 | 7724.00 | 7670.02 | 7825.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 7703.50 | 7671.22 | 7825.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 7830.15 | 7673.32 | 7823.87 | SL hit (close>static) qty=1.00 sl=7830.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 6685.00 | 6343.59 | 6342.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 6731.50 | 6365.79 | 6353.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 6656.00 | 6698.95 | 6562.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 6656.00 | 6698.95 | 6562.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 6563.00 | 6690.25 | 6566.75 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 6142.00 | 6499.81 | 6501.42 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 6845.00 | 6488.76 | 6488.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6901.50 | 6575.34 | 6537.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.19 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 5820.00 | 6551.66 | 6554.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 5794.00 | 6544.12 | 6551.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 5871.00 | 5846.71 | 6031.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:00:00 | 5871.00 | 5846.71 | 6031.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 5800.50 | 5838.52 | 6009.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 5771.00 | 5838.52 | 6009.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 5779.00 | 5833.07 | 6001.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 5782.50 | 5831.41 | 5997.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 5770.00 | 5830.76 | 5996.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5952.50 | 5808.63 | 5966.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 6119.00 | 5808.63 | 5966.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 5914.00 | 5809.68 | 5965.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 5965.00 | 5809.68 | 5965.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 5948.50 | 5812.21 | 5965.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 5948.50 | 5812.21 | 5965.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5482.45 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5490.05 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5493.38 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 5481.50 | 5750.69 | 5903.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 6136.00 | 5638.18 | 5781.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 6136.00 | 5638.18 | 5781.97 | SL hit (close>ema200) qty=0.50 sl=5638.18 alert=retest2 |

### Cycle 13 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 6378.00 | 5904.62 | 5903.66 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 5734.00 | 5927.99 | 5928.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 5720.00 | 5908.78 | 5918.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 5574.00 | 5553.83 | 5677.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 5534.00 | 5553.83 | 5677.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 5780.50 | 5554.19 | 5672.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 5872.00 | 5554.19 | 5672.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 5799.00 | 5556.62 | 5673.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 5708.00 | 5556.62 | 5673.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:15:00 | 5422.60 | 5560.43 | 5657.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 5565.00 | 5444.94 | 5556.33 | SL hit (close>ema200) qty=0.50 sl=5444.94 alert=retest2 |

### Cycle 15 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 5765.00 | 5608.26 | 5608.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 5793.50 | 5610.11 | 5609.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 5631.50 | 5654.26 | 5633.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 5630.50 | 5654.03 | 5633.22 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 5439.00 | 5617.81 | 5618.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 5420.00 | 5605.11 | 5612.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 12:15:00 | 5164.50 | 5143.11 | 5305.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 12:45:00 | 5169.40 | 5143.11 | 5305.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 5379.30 | 5150.76 | 5290.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 5379.30 | 5150.76 | 5290.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 5395.60 | 5153.20 | 5290.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 5395.60 | 5153.20 | 5290.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 5299.60 | 5235.26 | 5313.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 5294.80 | 5236.72 | 5313.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:15:00 | 5295.00 | 5236.72 | 5313.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 5443.80 | 5241.84 | 5314.69 | SL hit (close>static) qty=1.00 sl=5355.20 alert=retest2 |

### Cycle 17 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 5722.00 | 5369.69 | 5368.10 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-10 12:30:00 | 6696.20 | 2023-10-23 12:15:00 | 6620.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-10-10 13:00:00 | 6709.65 | 2023-10-23 12:15:00 | 6620.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-10-11 15:15:00 | 6705.55 | 2023-10-23 12:15:00 | 6620.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-10-12 09:45:00 | 6757.25 | 2023-10-23 12:15:00 | 6620.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-11-23 09:30:00 | 6715.00 | 2023-12-08 09:15:00 | 7386.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-23 12:15:00 | 6719.10 | 2023-12-08 09:15:00 | 7391.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-23 13:45:00 | 6705.00 | 2023-12-08 09:15:00 | 7375.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-24 09:15:00 | 6710.05 | 2023-12-08 09:15:00 | 7381.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-02 10:00:00 | 6126.85 | 2024-04-19 12:15:00 | 6284.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-04-02 14:00:00 | 6123.65 | 2024-04-19 12:15:00 | 6284.85 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-04-04 09:30:00 | 6125.00 | 2024-04-19 12:15:00 | 6284.85 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-04-18 14:45:00 | 6116.10 | 2024-04-19 12:15:00 | 6284.85 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-06-05 12:30:00 | 7093.60 | 2024-06-10 14:15:00 | 7802.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-09 13:45:00 | 7727.50 | 2024-12-10 11:15:00 | 7830.15 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-12-09 14:30:00 | 7724.00 | 2024-12-10 11:15:00 | 7830.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-12-10 09:15:00 | 7703.50 | 2024-12-10 11:15:00 | 7830.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-12-10 14:00:00 | 7720.00 | 2024-12-12 12:15:00 | 7844.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-13 09:30:00 | 7715.05 | 2024-12-20 12:15:00 | 7329.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 10:00:00 | 7712.35 | 2024-12-20 12:15:00 | 7326.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 14:00:00 | 7714.00 | 2024-12-20 12:15:00 | 7328.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 15:15:00 | 7698.00 | 2024-12-20 13:15:00 | 7313.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:30:00 | 7715.05 | 2024-12-26 09:15:00 | 7683.80 | STOP_HIT | 0.50 | 0.41% |
| SELL | retest2 | 2024-12-13 10:00:00 | 7712.35 | 2024-12-26 09:15:00 | 7683.80 | STOP_HIT | 0.50 | 0.37% |
| SELL | retest2 | 2024-12-17 14:00:00 | 7714.00 | 2024-12-26 09:15:00 | 7683.80 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2024-12-17 15:15:00 | 7698.00 | 2024-12-26 09:15:00 | 7683.80 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2025-03-26 09:15:00 | 6409.15 | 2025-03-28 13:15:00 | 6088.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 09:15:00 | 6409.15 | 2025-03-28 13:15:00 | 6089.45 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-03-26 10:45:00 | 6410.00 | 2025-03-28 13:15:00 | 6089.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 10:45:00 | 6410.00 | 2025-03-28 13:15:00 | 6089.45 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-15 11:30:00 | 6412.50 | 2025-05-07 15:15:00 | 6700.50 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-04-17 10:15:00 | 6407.50 | 2025-05-07 15:15:00 | 6700.50 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-09-23 10:15:00 | 5771.00 | 2025-10-09 09:15:00 | 5482.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 5779.00 | 2025-10-09 09:15:00 | 5490.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:15:00 | 5782.50 | 2025-10-09 09:15:00 | 5493.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-10-09 11:15:00 | 5481.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:15:00 | 5771.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.32% |
| SELL | retest2 | 2025-09-24 10:15:00 | 5779.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.18% |
| SELL | retest2 | 2025-09-24 13:15:00 | 5782.50 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.11% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.34% |
| SELL | retest2 | 2026-01-01 11:15:00 | 5708.00 | 2026-01-08 10:15:00 | 5422.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 11:15:00 | 5708.00 | 2026-01-27 15:15:00 | 5565.00 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2026-02-09 11:45:00 | 5760.00 | 2026-02-09 13:15:00 | 5835.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-09 12:30:00 | 5771.00 | 2026-02-09 13:15:00 | 5835.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-13 09:15:00 | 5749.00 | 2026-02-13 11:15:00 | 5765.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-04-24 12:30:00 | 5294.80 | 2026-04-27 09:15:00 | 5443.80 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-04-24 13:15:00 | 5295.00 | 2026-04-27 09:15:00 | 5443.80 | STOP_HIT | 1.00 | -2.81% |
