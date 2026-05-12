# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4497.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 44 |
| ALERT2 | 43 |
| ALERT2_SKIP | 21 |
| ALERT3 | 109 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 18 |
| TARGET_HIT | 6 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 29
- **Target hits / Stop hits / Partials:** 6 / 45 / 18
- **Avg / median % per leg:** 2.18% / 4.01%
- **Sum % (uncompounded):** 150.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 2 | 13 | 0 | 0.08% | 1.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 3 | 20.0% | 2 | 13 | 0 | 0.08% | 1.3% |
| SELL (all) | 54 | 37 | 68.5% | 4 | 32 | 18 | 2.76% | 148.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 37 | 68.5% | 4 | 32 | 18 | 2.76% | 148.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 40 | 58.0% | 6 | 45 | 18 | 2.18% | 150.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5986.00 | 5760.96 | 5737.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6052.00 | 5936.28 | 5848.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 6240.50 | 6273.72 | 6168.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 6240.50 | 6273.72 | 6168.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 6277.00 | 6352.26 | 6294.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 6277.00 | 6352.26 | 6294.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 6260.00 | 6333.81 | 6290.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 6112.00 | 6333.81 | 6290.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 6186.00 | 6278.84 | 6271.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 6173.50 | 6278.84 | 6271.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 6156.00 | 6254.27 | 6261.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 6085.00 | 6199.57 | 6233.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5952.00 | 5949.78 | 6024.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 5952.00 | 5949.78 | 6024.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 6065.00 | 5973.50 | 6022.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 6038.00 | 5973.50 | 6022.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 6011.50 | 5981.10 | 6021.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 5993.50 | 5981.10 | 6021.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 6017.00 | 6008.65 | 6007.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 6017.00 | 6008.65 | 6007.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 6075.50 | 6022.02 | 6013.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 5996.50 | 6025.72 | 6018.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 5998.00 | 6020.18 | 6016.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 5996.50 | 6020.18 | 6016.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 5974.00 | 6010.94 | 6012.69 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 6038.00 | 6002.41 | 6002.19 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 5971.00 | 6005.71 | 6008.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 5965.50 | 5986.16 | 5996.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 5776.00 | 5768.81 | 5826.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 5776.00 | 5768.81 | 5826.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5802.50 | 5765.41 | 5809.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 5861.50 | 5765.41 | 5809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 5750.00 | 5762.33 | 5804.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 5740.00 | 5757.56 | 5798.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 5738.50 | 5757.56 | 5798.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 5735.00 | 5707.69 | 5725.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5453.00 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5451.57 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5448.25 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 5505.00 | 5445.62 | 5500.05 | SL hit (close>ema200) qty=0.50 sl=5445.62 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 5602.50 | 5488.59 | 5486.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 5610.00 | 5512.88 | 5497.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5626.00 | 5674.80 | 5643.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5650.00 | 5669.84 | 5644.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 5619.00 | 5669.84 | 5644.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 5621.00 | 5660.07 | 5641.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 5621.00 | 5660.07 | 5641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 5600.00 | 5648.06 | 5638.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 5672.50 | 5648.06 | 5638.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 5753.00 | 5797.96 | 5799.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 5753.00 | 5797.96 | 5799.08 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 5869.50 | 5793.77 | 5792.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 5915.00 | 5846.33 | 5820.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 6098.50 | 6104.44 | 6046.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:30:00 | 6072.50 | 6104.44 | 6046.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 6155.00 | 6204.49 | 6167.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 6214.00 | 6204.49 | 6167.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 6206.50 | 6204.89 | 6170.65 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 6105.50 | 6169.68 | 6173.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 6082.00 | 6152.14 | 6164.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 6221.50 | 6090.02 | 6118.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 6146.50 | 6101.32 | 6120.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 6127.00 | 6124.60 | 6128.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 6175.00 | 6134.81 | 6131.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 6175.00 | 6134.81 | 6131.96 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 6038.50 | 6123.70 | 6134.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 6020.00 | 6102.96 | 6123.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 6026.00 | 6018.85 | 6052.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 6026.00 | 6018.85 | 6052.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 6049.00 | 6025.07 | 6049.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 6074.50 | 6025.07 | 6049.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 6053.00 | 6030.65 | 6049.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 6053.00 | 6030.65 | 6049.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 6002.00 | 6024.92 | 6045.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 5985.00 | 6024.92 | 6045.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:15:00 | 5963.00 | 6020.34 | 6041.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:45:00 | 5997.00 | 5998.48 | 6020.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 5991.00 | 5981.61 | 5999.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5982.50 | 5981.79 | 5998.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 5972.50 | 5981.79 | 5998.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5685.75 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5697.15 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5691.45 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 5664.85 | 5715.49 | 5760.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 5673.88 | 5715.49 | 5760.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 5741.00 | 5714.60 | 5752.11 | SL hit (close>ema200) qty=0.50 sl=5714.60 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 5612.00 | 5602.36 | 5601.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 5662.50 | 5614.39 | 5607.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 6306.50 | 6313.40 | 6137.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 6306.50 | 6313.40 | 6137.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 6080.50 | 6293.20 | 6266.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 6089.00 | 6293.20 | 6266.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 6051.00 | 6244.76 | 6246.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 6003.00 | 6076.74 | 6144.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6085.00 | 6055.44 | 6110.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 6085.00 | 6055.44 | 6110.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 5888.00 | 6028.28 | 6088.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 5874.50 | 5943.38 | 6021.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 6115.50 | 5929.62 | 5914.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 6115.50 | 5929.62 | 5914.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 6294.00 | 6135.52 | 6062.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 6249.00 | 6274.49 | 6210.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 6249.00 | 6274.49 | 6210.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 6263.00 | 6271.54 | 6225.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 6301.00 | 6271.54 | 6225.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 6182.50 | 6214.05 | 6217.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 6182.50 | 6214.05 | 6217.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 6139.50 | 6199.14 | 6209.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 6202.00 | 6169.94 | 6188.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 6227.00 | 6181.35 | 6191.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 6227.00 | 6181.35 | 6191.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 6247.50 | 6203.76 | 6200.16 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 6180.00 | 6217.40 | 6218.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 6124.00 | 6198.72 | 6209.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 6207.00 | 6196.33 | 6205.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 6212.00 | 6199.46 | 6205.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 6212.00 | 6199.46 | 6205.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 6192.00 | 6197.97 | 6204.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 6208.00 | 6197.97 | 6204.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 6155.00 | 6189.38 | 6200.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 6205.50 | 6189.38 | 6200.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 6146.50 | 6180.80 | 6195.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 6123.50 | 6164.33 | 6182.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 6355.00 | 6189.86 | 6188.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 6355.00 | 6189.86 | 6188.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 6441.50 | 6240.19 | 6211.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 7077.50 | 7079.99 | 7010.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:45:00 | 7076.00 | 7079.99 | 7010.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 7227.00 | 7147.23 | 7103.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 7253.50 | 7174.83 | 7158.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:00:00 | 7260.00 | 7198.90 | 7175.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 7305.00 | 7231.94 | 7211.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 7141.00 | 7199.74 | 7205.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 7141.00 | 7199.74 | 7205.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 7075.00 | 7148.92 | 7173.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 7250.00 | 7193.12 | 7185.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 7450.00 | 7264.60 | 7223.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 7499.50 | 7511.12 | 7422.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 7499.50 | 7511.12 | 7422.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 7463.00 | 7498.91 | 7431.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 7415.50 | 7498.91 | 7431.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 7423.00 | 7483.73 | 7431.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 7423.00 | 7483.73 | 7431.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 7381.50 | 7463.29 | 7426.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 7378.00 | 7463.29 | 7426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 7386.00 | 7447.83 | 7422.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 7355.50 | 7447.83 | 7422.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 7330.50 | 7424.36 | 7414.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 7305.00 | 7424.36 | 7414.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 7340.00 | 7394.71 | 7401.97 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 7461.50 | 7408.07 | 7407.38 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 7212.00 | 7368.86 | 7389.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 7165.50 | 7328.18 | 7369.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 7153.00 | 7093.39 | 7167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 7174.50 | 7116.03 | 7165.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 7207.00 | 7116.03 | 7165.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 7157.50 | 7124.32 | 7164.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 7174.50 | 7124.32 | 7164.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 7158.00 | 7131.06 | 7163.95 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 7215.00 | 7181.41 | 7177.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 7277.50 | 7206.88 | 7190.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 7208.50 | 7221.82 | 7202.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 7233.00 | 7224.06 | 7205.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 7259.00 | 7224.06 | 7205.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 7247.50 | 7376.20 | 7382.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 7247.50 | 7376.20 | 7382.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 7184.00 | 7337.76 | 7364.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 6950.50 | 6886.46 | 6989.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 6932.00 | 6897.50 | 6944.82 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 7035.00 | 6961.51 | 6959.03 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 6960.00 | 6990.57 | 6991.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 6916.00 | 6975.65 | 6984.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 6713.00 | 6708.83 | 6767.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:15:00 | 6708.00 | 6708.83 | 6767.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 6735.00 | 6714.47 | 6748.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 6735.00 | 6714.47 | 6748.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 6735.00 | 6718.57 | 6747.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 6925.00 | 6718.57 | 6747.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 6885.00 | 6751.86 | 6759.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 6973.50 | 6751.86 | 6759.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 6951.00 | 6791.69 | 6776.97 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 6807.00 | 6823.96 | 6825.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 6763.50 | 6811.87 | 6820.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 6698.00 | 6678.09 | 6713.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 6686.50 | 6652.25 | 6688.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 6686.50 | 6652.25 | 6688.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 6678.00 | 6657.40 | 6687.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 6599.00 | 6657.40 | 6687.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6534.50 | 6632.82 | 6673.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 6473.50 | 6600.95 | 6655.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 6433.00 | 6406.50 | 6403.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 6433.00 | 6406.50 | 6403.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 6466.50 | 6427.14 | 6414.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 6437.00 | 6439.27 | 6426.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:00:00 | 6437.00 | 6439.27 | 6426.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 6437.50 | 6437.83 | 6427.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:45:00 | 6431.50 | 6437.83 | 6427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 6423.00 | 6434.87 | 6427.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 6457.50 | 6434.87 | 6427.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 6434.00 | 6434.69 | 6427.90 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 6402.50 | 6421.78 | 6423.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 6387.50 | 6414.92 | 6420.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 5978.00 | 5947.59 | 6035.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 5978.00 | 5947.59 | 6035.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 5970.00 | 5953.82 | 6023.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:15:00 | 5957.00 | 5953.82 | 6023.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 5958.00 | 5974.86 | 5999.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:15:00 | 5920.00 | 5974.86 | 5999.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:30:00 | 5955.50 | 5962.27 | 5988.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5845.00 | 5813.67 | 5854.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 5822.50 | 5815.44 | 5851.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 5806.00 | 5811.68 | 5843.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5659.15 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5660.10 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5624.00 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5657.72 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:15:00 | 5531.38 | 5589.65 | 5678.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:15:00 | 5515.70 | 5589.65 | 5678.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-01 11:15:00 | 5361.30 | 5466.43 | 5558.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 4241.50 | 4142.50 | 4134.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 4267.50 | 4182.78 | 4155.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 4198.50 | 4233.76 | 4192.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 4203.50 | 4229.15 | 4200.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 4203.50 | 4229.15 | 4200.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 4195.50 | 4222.42 | 4200.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 4181.00 | 4222.42 | 4200.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 4144.50 | 4206.84 | 4195.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 4144.50 | 4206.84 | 4195.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 4173.00 | 4200.07 | 4193.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 4181.50 | 4204.76 | 4196.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 4186.50 | 4197.32 | 4194.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:45:00 | 4198.50 | 4194.51 | 4193.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 4192.50 | 4194.01 | 4193.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 4174.00 | 4190.00 | 4191.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4174.00 | 4190.00 | 4191.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 4159.50 | 4183.90 | 4188.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 4095.00 | 4073.99 | 4104.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 4121.00 | 4083.39 | 4105.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 4140.00 | 4083.39 | 4105.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 4144.00 | 4095.51 | 4109.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 4144.00 | 4095.51 | 4109.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 4181.00 | 4126.53 | 4121.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 4206.00 | 4142.42 | 4129.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 4188.50 | 4193.43 | 4167.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 4161.50 | 4193.43 | 4167.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 4168.00 | 4188.34 | 4167.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 4159.00 | 4188.34 | 4167.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 4159.50 | 4182.58 | 4166.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 4157.00 | 4182.58 | 4166.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 4155.00 | 4177.06 | 4165.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 4156.50 | 4177.06 | 4165.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 4152.00 | 4168.28 | 4163.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 4152.00 | 4168.28 | 4163.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 4138.00 | 4157.86 | 4159.11 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 4171.00 | 4161.47 | 4160.60 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 4153.00 | 4159.05 | 4159.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 4117.50 | 4150.74 | 4155.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 3944.80 | 3977.03 | 3993.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 3935.80 | 3965.08 | 3984.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 3944.50 | 3950.54 | 3959.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 4069.90 | 3984.58 | 3973.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 4069.90 | 3984.58 | 3973.72 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 3942.00 | 3975.11 | 3975.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 3743.00 | 3923.85 | 3951.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 3688.20 | 3674.32 | 3724.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 3682.40 | 3674.32 | 3724.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3690.00 | 3685.24 | 3714.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 3675.00 | 3684.19 | 3711.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 3669.70 | 3681.29 | 3707.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 3667.60 | 3676.40 | 3702.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 3740.80 | 3692.88 | 3701.92 | SL hit (close>static) qty=1.00 sl=3738.80 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 3486.60 | 3435.23 | 3428.88 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3368.50 | 3415.37 | 3420.55 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 3547.70 | 3426.41 | 3419.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 3619.10 | 3502.08 | 3463.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 3476.60 | 3533.63 | 3501.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3382.80 | 3503.46 | 3490.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 3382.80 | 3503.46 | 3490.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 3389.20 | 3469.18 | 3476.66 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 3625.70 | 3495.51 | 3484.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 3747.50 | 3657.21 | 3593.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 3620.10 | 3692.97 | 3635.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3593.90 | 3673.16 | 3631.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 3627.10 | 3662.17 | 3630.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 3630.70 | 3649.07 | 3629.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 3626.40 | 3643.25 | 3629.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 3493.00 | 3606.12 | 3614.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 3493.00 | 3606.12 | 3614.20 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 3707.00 | 3610.55 | 3608.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3836.00 | 3672.03 | 3638.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 4083.90 | 4091.72 | 3999.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:00:00 | 4083.90 | 4091.72 | 3999.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3943.00 | 4060.83 | 4030.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 3952.80 | 4060.83 | 4030.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 3916.40 | 4031.95 | 4019.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 3916.40 | 4031.95 | 4019.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 3969.50 | 4005.32 | 4009.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 3936.00 | 3972.39 | 3985.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 3951.00 | 3950.15 | 3969.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 3951.00 | 3950.15 | 3969.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 3968.00 | 3953.72 | 3968.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 3994.30 | 3953.72 | 3968.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3927.30 | 3948.43 | 3965.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 3917.40 | 3948.43 | 3965.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:00:00 | 3919.00 | 3942.55 | 3960.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 4020.50 | 3967.32 | 3968.44 | SL hit (close>static) qty=1.00 sl=4019.80 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 4058.60 | 3985.58 | 3976.64 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 3922.30 | 3972.45 | 3974.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 3874.00 | 3939.85 | 3957.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 3904.30 | 3904.17 | 3932.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 3904.30 | 3904.17 | 3932.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 3833.10 | 3803.16 | 3833.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 3833.10 | 3803.16 | 3833.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 3860.00 | 3814.53 | 3836.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 3893.50 | 3814.53 | 3836.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 3890.60 | 3829.74 | 3841.00 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3961.00 | 3855.99 | 3851.91 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3860.00 | 3887.34 | 3889.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 3846.20 | 3879.11 | 3885.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 3699.20 | 3692.17 | 3746.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 3699.20 | 3692.17 | 3746.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3661.80 | 3687.51 | 3735.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 3651.00 | 3679.55 | 3727.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 3803.00 | 3712.21 | 3728.11 | SL hit (close>static) qty=1.00 sl=3754.90 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 3796.00 | 3743.97 | 3740.72 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3623.80 | 3731.79 | 3739.91 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 3759.00 | 3730.51 | 3727.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3837.90 | 3764.50 | 3744.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 3796.30 | 3797.39 | 3769.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:45:00 | 3797.70 | 3797.39 | 3769.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3768.10 | 3791.53 | 3769.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 3768.10 | 3791.53 | 3769.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 3744.70 | 3782.17 | 3767.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 3744.70 | 3782.17 | 3767.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3719.30 | 3769.59 | 3762.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 3719.30 | 3769.59 | 3762.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 3715.10 | 3758.70 | 3758.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 3613.60 | 3758.70 | 3758.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 3594.20 | 3725.80 | 3743.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 3550.10 | 3674.41 | 3706.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3562.30 | 3524.48 | 3573.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3628.20 | 3547.70 | 3575.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3628.20 | 3547.70 | 3575.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3618.80 | 3561.92 | 3579.70 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 3633.60 | 3592.45 | 3590.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3677.00 | 3622.67 | 3606.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 3626.60 | 3672.14 | 3647.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3624.30 | 3662.57 | 3645.21 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 3580.00 | 3632.31 | 3634.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 3565.30 | 3618.91 | 3628.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 3665.00 | 3621.74 | 3627.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 3687.30 | 3634.85 | 3632.97 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 3623.70 | 3634.35 | 3634.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3472.70 | 3602.02 | 3619.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 3486.40 | 3455.68 | 3506.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3542.70 | 3473.08 | 3509.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 3545.50 | 3473.08 | 3509.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 3512.00 | 3480.86 | 3509.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 3537.50 | 3480.86 | 3509.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3513.70 | 3487.43 | 3510.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 3517.00 | 3487.43 | 3510.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3518.00 | 3493.55 | 3511.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3632.00 | 3493.55 | 3511.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3729.80 | 3540.80 | 3530.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3756.60 | 3583.96 | 3551.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3618.90 | 3664.87 | 3616.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3609.90 | 3653.88 | 3615.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3609.90 | 3653.88 | 3615.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 3647.80 | 3652.66 | 3618.84 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 3498.00 | 3593.73 | 3603.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 3478.60 | 3544.76 | 3576.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 3570.00 | 3537.14 | 3559.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 3538.00 | 3557.20 | 3563.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 3391.50 | 3522.99 | 3547.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 3497.70 | 3487.41 | 3519.21 | SL hit (close>ema200) qty=0.50 sl=3487.41 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 3610.70 | 3548.74 | 3540.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 3645.00 | 3567.99 | 3549.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 3834.40 | 3847.73 | 3771.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 3828.40 | 3847.73 | 3771.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3816.90 | 3870.43 | 3845.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 3860.00 | 3866.60 | 3845.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 3855.00 | 3866.60 | 3845.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 4246.00 | 4151.84 | 4064.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 4263.00 | 4334.89 | 4335.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 4233.00 | 4288.02 | 4310.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 4166.00 | 4147.08 | 4192.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:45:00 | 4160.00 | 4147.08 | 4192.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 4191.90 | 4159.76 | 4190.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 4191.90 | 4159.76 | 4190.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 4121.80 | 4152.17 | 4184.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 4039.90 | 4138.95 | 4172.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4185.00 | 4099.65 | 4098.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 4185.00 | 4099.65 | 4098.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 4226.60 | 4168.78 | 4139.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 11:15:00 | 5993.50 | 2025-05-23 15:15:00 | 6017.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-06-05 11:30:00 | 5740.00 | 2025-06-13 09:15:00 | 5453.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 12:00:00 | 5738.50 | 2025-06-13 09:15:00 | 5451.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 09:30:00 | 5735.00 | 2025-06-13 09:15:00 | 5448.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 11:30:00 | 5740.00 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-06-05 12:00:00 | 5738.50 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-06-09 09:30:00 | 5735.00 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2025-06-20 09:15:00 | 5672.50 | 2025-06-26 10:15:00 | 5753.00 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2025-07-09 13:15:00 | 6127.00 | 2025-07-10 09:15:00 | 6175.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-15 12:15:00 | 5985.00 | 2025-07-24 12:15:00 | 5685.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 13:15:00 | 5963.00 | 2025-07-24 12:15:00 | 5697.15 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2025-07-16 10:45:00 | 5997.00 | 2025-07-24 12:15:00 | 5691.45 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-07-17 10:00:00 | 5991.00 | 2025-07-24 14:15:00 | 5664.85 | PARTIAL | 0.50 | 5.44% |
| SELL | retest2 | 2025-07-17 11:15:00 | 5972.50 | 2025-07-24 14:15:00 | 5673.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 12:15:00 | 5985.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-07-15 13:15:00 | 5963.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-07-16 10:45:00 | 5997.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-07-17 10:00:00 | 5991.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-07-17 11:15:00 | 5972.50 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-08-08 14:30:00 | 5874.50 | 2025-08-13 11:15:00 | 6115.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-08-20 10:15:00 | 6301.00 | 2025-08-21 13:15:00 | 6182.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-08-29 14:15:00 | 6123.50 | 2025-09-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-09-17 10:45:00 | 7253.50 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-17 15:00:00 | 7260.00 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-19 09:15:00 | 7305.00 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-06 12:15:00 | 7259.00 | 2025-10-08 13:15:00 | 7247.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-06 11:00:00 | 6473.50 | 2025-11-11 10:15:00 | 6433.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-11-20 10:15:00 | 5957.00 | 2025-11-27 09:15:00 | 5659.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 10:45:00 | 5958.00 | 2025-11-27 09:15:00 | 5660.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 11:15:00 | 5920.00 | 2025-11-27 09:15:00 | 5624.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 12:30:00 | 5955.50 | 2025-11-27 09:15:00 | 5657.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:00:00 | 5822.50 | 2025-11-28 10:15:00 | 5531.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:30:00 | 5806.00 | 2025-11-28 10:15:00 | 5515.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:15:00 | 5957.00 | 2025-12-01 11:15:00 | 5361.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 10:45:00 | 5958.00 | 2025-12-01 11:15:00 | 5362.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 11:15:00 | 5920.00 | 2025-12-01 11:15:00 | 5359.95 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2025-11-21 12:30:00 | 5955.50 | 2025-12-01 12:15:00 | 5328.00 | TARGET_HIT | 0.50 | 10.54% |
| SELL | retest2 | 2025-11-26 11:00:00 | 5822.50 | 2025-12-02 09:15:00 | 5428.00 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2025-11-26 12:30:00 | 5806.00 | 2025-12-02 09:15:00 | 5428.00 | STOP_HIT | 0.50 | 6.51% |
| BUY | retest2 | 2025-12-16 11:30:00 | 4181.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-12-16 15:00:00 | 4186.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-12-17 10:45:00 | 4198.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-17 11:30:00 | 4192.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-01-01 12:00:00 | 3944.80 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-01-01 13:30:00 | 3935.80 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-01-05 09:45:00 | 3944.50 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2026-01-13 10:45:00 | 3675.00 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3669.70 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-01-13 12:45:00 | 3667.60 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 3673.00 | 2026-01-20 13:15:00 | 3489.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 3675.00 | 2026-01-20 13:15:00 | 3491.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3667.00 | 2026-01-20 13:15:00 | 3483.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 3673.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2026-01-16 11:45:00 | 3675.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3667.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.15% |
| BUY | retest2 | 2026-02-05 12:15:00 | 3627.10 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2026-02-05 13:30:00 | 3630.70 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-02-05 14:45:00 | 3626.40 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-02-18 10:15:00 | 3917.40 | 2026-02-18 13:15:00 | 4020.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-02-18 11:00:00 | 3919.00 | 2026-02-18 13:15:00 | 4020.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-03-05 10:30:00 | 3651.00 | 2026-03-05 14:15:00 | 3803.00 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2026-04-01 10:45:00 | 3570.00 | 2026-04-02 09:15:00 | 3391.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 3570.00 | 2026-04-02 13:15:00 | 3497.70 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2026-04-01 14:30:00 | 3538.00 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-06 10:15:00 | 3568.00 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-06 11:00:00 | 3562.80 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-04-13 10:45:00 | 3860.00 | 2026-04-17 09:15:00 | 4246.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:15:00 | 3855.00 | 2026-04-17 09:15:00 | 4240.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 4039.90 | 2026-05-05 09:15:00 | 4185.00 | STOP_HIT | 1.00 | -3.59% |
