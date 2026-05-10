# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 7010.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 35
- **Target hits / Stop hits / Partials:** 0 / 45 / 7
- **Avg / median % per leg:** -0.03% / -0.82%
- **Sum % (uncompounded):** -1.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.62% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.62% | -2.5% |
| SELL (all) | 48 | 17 | 35.4% | 0 | 41 | 7 | 0.02% | 0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 17 | 35.4% | 0 | 41 | 7 | 0.02% | 0.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 17 | 32.7% | 0 | 45 | 7 | -0.03% | -1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 6040.00 | 5617.45 | 5615.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 6047.00 | 5702.36 | 5660.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5885.00 | 5922.32 | 5816.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 5885.00 | 5922.32 | 5816.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5897.50 | 5956.52 | 5864.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 5870.50 | 5956.52 | 5864.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 5875.00 | 5951.70 | 5865.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 5871.00 | 5951.70 | 5865.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 5866.50 | 5950.85 | 5865.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 5888.00 | 5950.85 | 5865.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 5859.50 | 5948.57 | 5866.03 | SL hit (close<static) qty=1.00 sl=5862.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 5880.00 | 5931.16 | 5864.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 5888.00 | 5930.73 | 5864.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 5877.50 | 5930.73 | 5864.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 5896.50 | 5929.57 | 5865.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 5895.00 | 5929.57 | 5865.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 5843.00 | 5927.84 | 5865.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 5843.00 | 5927.84 | 5865.32 | SL hit (close<static) qty=1.00 sl=5862.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 5843.00 | 5927.84 | 5865.32 | SL hit (close<static) qty=1.00 sl=5862.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 5843.00 | 5927.84 | 5865.32 | SL hit (close<static) qty=1.00 sl=5862.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 5843.00 | 5927.84 | 5865.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 5842.50 | 5926.99 | 5865.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 5842.50 | 5926.99 | 5865.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 5819.50 | 5924.07 | 5865.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 5819.50 | 5924.07 | 5865.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 5792.00 | 5922.76 | 5864.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 5792.00 | 5922.76 | 5864.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 5837.00 | 5827.40 | 5823.50 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 5687.50 | 5819.20 | 5819.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5663.00 | 5813.26 | 5816.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 5233.30 | 5204.92 | 5363.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 5233.30 | 5204.92 | 5363.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5360.80 | 5210.63 | 5355.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 5359.80 | 5210.63 | 5355.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 5370.50 | 5212.22 | 5355.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 5370.50 | 5212.22 | 5355.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 5366.60 | 5213.76 | 5355.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 5340.00 | 5215.02 | 5355.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 5348.70 | 5217.61 | 5355.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 5343.60 | 5218.98 | 5355.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 5340.60 | 5220.19 | 5355.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 5345.00 | 5221.43 | 5355.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 5339.10 | 5221.43 | 5355.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5335.50 | 5222.57 | 5354.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5358.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5358.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 5339.20 | 5232.14 | 5355.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 5390.90 | 5233.72 | 5355.44 | SL hit (close>static) qty=1.00 sl=5358.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 5315.50 | 5272.29 | 5362.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5316.50 | 5244.11 | 5322.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 12:30:00 | 5227.00 | 5244.79 | 5321.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 14:30:00 | 5228.00 | 5244.53 | 5320.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 5224.50 | 5213.08 | 5288.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 5227.00 | 5212.23 | 5282.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 5227.50 | 5212.45 | 5273.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 5211.50 | 5212.55 | 5272.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 5216.50 | 5212.26 | 5272.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 5222.00 | 5210.90 | 5270.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 5294.00 | 5212.31 | 5270.26 | SL hit (close>static) qty=1.00 sl=5275.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 5294.00 | 5212.31 | 5270.26 | SL hit (close>static) qty=1.00 sl=5275.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 5294.00 | 5212.31 | 5270.26 | SL hit (close>static) qty=1.00 sl=5275.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:45:00 | 5216.50 | 5220.57 | 5268.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 5252.00 | 5221.96 | 5268.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 5252.00 | 5221.96 | 5268.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 5258.00 | 5223.36 | 5267.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 5258.00 | 5223.36 | 5267.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 5265.50 | 5223.78 | 5267.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 5265.50 | 5223.78 | 5267.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 5229.50 | 5223.84 | 5267.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 5048.50 | 5224.40 | 5266.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 5049.72 | 5222.89 | 5265.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4965.65 | 5220.41 | 5264.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4966.60 | 5220.41 | 5264.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4963.27 | 5220.41 | 5264.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4965.65 | 5220.41 | 5264.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4955.68 | 5207.59 | 5256.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 5130.50 | 5117.31 | 5192.35 | SL hit (close>ema200) qty=0.50 sl=5117.31 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:15:00 | 5190.00 | 5114.21 | 5179.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 5195.50 | 5116.77 | 5179.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:15:00 | 5192.50 | 5124.94 | 5181.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5174.00 | 5126.55 | 5181.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 5167.00 | 5128.05 | 5181.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 5163.50 | 5128.35 | 5181.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 5185.00 | 5129.35 | 5181.02 | SL hit (close>static) qty=1.00 sl=5182.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 5185.00 | 5129.35 | 5181.02 | SL hit (close>static) qty=1.00 sl=5182.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 5169.00 | 5129.99 | 5181.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 5186.00 | 5130.54 | 5181.11 | SL hit (close>static) qty=1.00 sl=5182.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:45:00 | 5168.50 | 5132.01 | 5181.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 5174.00 | 5132.43 | 5181.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 5174.00 | 5132.43 | 5181.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 5193.50 | 5133.04 | 5181.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 5193.50 | 5133.04 | 5181.13 | SL hit (close>static) qty=1.00 sl=5182.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-02 13:45:00 | 5201.00 | 5133.04 | 5181.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 5185.50 | 5133.56 | 5181.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 5183.00 | 5133.56 | 5181.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 5165.00 | 5133.87 | 5181.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 5153.50 | 5133.87 | 5181.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 5163.00 | 5133.89 | 5178.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 5161.50 | 5134.84 | 5178.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 5159.00 | 5135.53 | 5178.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 5180.00 | 5136.74 | 5177.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 5163.00 | 5136.74 | 5177.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 5121.00 | 5136.58 | 5177.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:30:00 | 5108.50 | 5136.03 | 5176.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:45:00 | 5113.00 | 5131.08 | 5172.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 5108.00 | 5130.97 | 5172.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:45:00 | 5114.00 | 5130.71 | 5171.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 5150.00 | 5130.90 | 5171.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 5176.00 | 5130.90 | 5171.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5192.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5192.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5192.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5192.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5194.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5194.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5194.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 5237.50 | 5133.15 | 5171.54 | SL hit (close>static) qty=1.00 sl=5194.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 5245.00 | 5133.15 | 5171.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 5209.50 | 5133.91 | 5171.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5282.50 | 5145.82 | 5175.81 | SL hit (close>static) qty=1.00 sl=5280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5282.50 | 5145.82 | 5175.81 | SL hit (close>static) qty=1.00 sl=5280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5282.50 | 5145.82 | 5175.81 | SL hit (close>static) qty=1.00 sl=5280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 5282.50 | 5145.82 | 5175.81 | SL hit (close>static) qty=1.00 sl=5280.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 5194.00 | 5163.94 | 5182.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 5195.00 | 5160.55 | 5179.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 5186.50 | 5160.83 | 5179.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 5175.50 | 5163.52 | 5179.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 5214.00 | 5164.03 | 5179.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 5250.00 | 5164.03 | 5179.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 5243.00 | 5165.53 | 5180.42 | SL hit (close>static) qty=1.00 sl=5242.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 5243.00 | 5165.53 | 5180.42 | SL hit (close>static) qty=1.00 sl=5242.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 5243.00 | 5165.53 | 5180.42 | SL hit (close>static) qty=1.00 sl=5242.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 5243.00 | 5165.53 | 5180.42 | SL hit (close>static) qty=1.00 sl=5242.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 5192.50 | 5171.11 | 5182.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 5193.00 | 5171.11 | 5182.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5190.00 | 5171.30 | 5182.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 5179.00 | 5171.42 | 5182.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 5172.00 | 5171.63 | 5182.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 5179.00 | 5171.63 | 5182.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:00:00 | 5176.50 | 5171.68 | 5182.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 5161.50 | 5168.84 | 5180.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 5219.00 | 5170.29 | 5180.10 | SL hit (close>static) qty=1.00 sl=5218.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 5219.00 | 5170.29 | 5180.10 | SL hit (close>static) qty=1.00 sl=5218.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 5219.00 | 5170.29 | 5180.10 | SL hit (close>static) qty=1.00 sl=5218.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 5219.00 | 5170.29 | 5180.10 | SL hit (close>static) qty=1.00 sl=5218.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 5087.50 | 5180.88 | 5184.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 4833.12 | 5100.43 | 5140.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 12:15:00 | 5036.00 | 4987.42 | 5070.98 | SL hit (close>ema200) qty=0.50 sl=4987.42 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 5662.00 | 5141.64 | 5139.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 5830.00 | 5148.49 | 5143.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 12:15:00 | 5997.00 | 6017.35 | 5770.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:00:00 | 5997.00 | 6017.35 | 5770.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-04 09:15:00 | 5888.00 | 2025-07-04 11:15:00 | 5859.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-09 09:45:00 | 5880.00 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-09 11:00:00 | 5888.00 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-09 11:30:00 | 5877.50 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-15 15:00:00 | 5340.00 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-16 09:45:00 | 5348.70 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-16 10:30:00 | 5343.60 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-16 12:00:00 | 5340.60 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-16 13:15:00 | 5339.10 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-16 14:00:00 | 5335.50 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-17 14:00:00 | 5339.20 | 2025-09-17 14:15:00 | 5390.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-23 09:45:00 | 5315.50 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-07 12:30:00 | 5227.00 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-07 14:30:00 | 5228.00 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-16 10:00:00 | 5224.50 | 2025-11-07 09:15:00 | 5049.72 | PARTIAL | 0.50 | 3.35% |
| SELL | retest2 | 2025-10-20 10:45:00 | 5227.00 | 2025-11-07 10:15:00 | 4965.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 10:45:00 | 5211.50 | 2025-11-07 10:15:00 | 4966.60 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-10-28 11:30:00 | 5216.50 | 2025-11-07 10:15:00 | 4963.27 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-10-29 09:30:00 | 5222.00 | 2025-11-07 10:15:00 | 4965.65 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-11-03 11:45:00 | 5216.50 | 2025-11-10 09:15:00 | 4955.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:00:00 | 5224.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2025-10-20 10:45:00 | 5227.00 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2025-10-28 10:45:00 | 5211.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-10-28 11:30:00 | 5216.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-10-29 09:30:00 | 5222.00 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2025-11-03 11:45:00 | 5216.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-11-07 09:15:00 | 5048.50 | 2025-12-01 13:15:00 | 5185.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-11-26 15:15:00 | 5190.00 | 2025-12-01 13:15:00 | 5185.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-11-27 10:30:00 | 5195.50 | 2025-12-01 15:15:00 | 5186.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-11-28 12:15:00 | 5192.50 | 2025-12-02 13:15:00 | 5193.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-01 10:45:00 | 5167.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-01 11:45:00 | 5163.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-01 14:45:00 | 5169.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-02 11:45:00 | 5168.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-03 09:15:00 | 5153.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-12-04 13:30:00 | 5163.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-05 09:30:00 | 5161.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-05 12:15:00 | 5159.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-08 12:30:00 | 5108.50 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-12-09 13:45:00 | 5113.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-12-09 15:15:00 | 5108.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-12-10 09:45:00 | 5114.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-17 12:15:00 | 5194.00 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-22 11:45:00 | 5195.00 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-22 12:30:00 | 5186.50 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-23 14:30:00 | 5175.50 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-12-29 11:15:00 | 5179.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 12:30:00 | 5172.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 13:00:00 | 5179.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 14:00:00 | 5176.50 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-08 14:45:00 | 5087.50 | 2026-01-20 09:15:00 | 4833.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:45:00 | 5087.50 | 2026-01-28 12:15:00 | 5036.00 | STOP_HIT | 0.50 | 1.01% |
