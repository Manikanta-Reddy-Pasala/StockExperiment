# Shree Cement Ltd. (SHREECEM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 25445.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 98 |
| ALERT2 | 94 |
| ALERT2_SKIP | 42 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 129 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 48 / 83
- **Target hits / Stop hits / Partials:** 3 / 124 / 4
- **Avg / median % per leg:** 0.29% / -0.63%
- **Sum % (uncompounded):** 38.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 16 | 24.6% | 2 | 63 | 0 | 0.18% | 11.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 65 | 16 | 24.6% | 2 | 63 | 0 | 0.18% | 11.5% |
| SELL (all) | 66 | 32 | 48.5% | 1 | 61 | 4 | 0.41% | 26.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 64 | 30 | 46.9% | 0 | 61 | 3 | 0.19% | 11.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 129 | 46 | 35.7% | 2 | 124 | 3 | 0.18% | 23.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 25531.65 | 25854.32 | 25869.74 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 13:15:00 | 26125.80 | 25840.77 | 25835.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 14:15:00 | 26328.00 | 25938.22 | 25879.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 25937.45 | 26069.43 | 25983.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 25937.45 | 26069.43 | 25983.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 25937.45 | 26069.43 | 25983.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 25874.35 | 26069.43 | 25983.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 25922.25 | 26039.99 | 25977.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 25921.80 | 26039.99 | 25977.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 25830.05 | 25958.82 | 25949.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:15:00 | 25832.00 | 25958.82 | 25949.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 25776.45 | 25922.35 | 25933.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 25564.00 | 25830.36 | 25886.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 25620.35 | 25587.49 | 25682.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 13:45:00 | 25619.75 | 25587.49 | 25682.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 25668.10 | 25603.61 | 25681.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 25668.10 | 25603.61 | 25681.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 25600.70 | 25603.03 | 25673.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 25672.35 | 25596.61 | 25664.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 25632.15 | 25603.72 | 25661.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 25632.15 | 25603.72 | 25661.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 25527.05 | 25588.39 | 25649.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:30:00 | 25640.10 | 25588.39 | 25649.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 25519.95 | 25467.64 | 25519.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 25519.95 | 25467.64 | 25519.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 25650.00 | 25504.12 | 25531.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 25650.00 | 25504.12 | 25531.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 25712.20 | 25545.73 | 25547.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 25627.40 | 25545.73 | 25547.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 12:15:00 | 25620.00 | 25560.59 | 25554.42 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 25315.25 | 25541.65 | 25551.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 25250.00 | 25407.15 | 25472.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 25140.00 | 25094.28 | 25236.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 25299.45 | 25094.28 | 25236.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 25137.60 | 25102.94 | 25227.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 25028.35 | 25102.94 | 25227.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:45:00 | 25002.60 | 25074.64 | 25192.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:30:00 | 25014.20 | 25059.71 | 25174.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 25477.05 | 25017.63 | 25106.68 | SL hit (close>static) qty=1.00 sl=25425.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 25455.00 | 25162.31 | 25160.54 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 24730.05 | 25163.05 | 25181.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 24356.25 | 25001.69 | 25106.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 24920.10 | 24872.55 | 25011.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 14:45:00 | 24858.90 | 24872.55 | 25011.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 24801.10 | 24858.26 | 24992.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 25219.05 | 24858.26 | 24992.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 25270.00 | 24940.61 | 25017.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:30:00 | 25122.35 | 24940.61 | 25017.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 25354.85 | 25023.46 | 25048.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 25354.85 | 25023.46 | 25048.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 25419.85 | 25102.73 | 25081.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 25472.95 | 25270.56 | 25172.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 27255.00 | 27364.34 | 27135.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 27255.00 | 27364.34 | 27135.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 27380.15 | 27477.75 | 27386.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 27318.60 | 27477.75 | 27386.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 27339.95 | 27450.19 | 27382.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 27299.85 | 27450.19 | 27382.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 27332.65 | 27426.68 | 27378.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:15:00 | 27312.05 | 27426.68 | 27378.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 27428.30 | 27427.01 | 27382.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 27548.95 | 27416.46 | 27384.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 27194.05 | 27393.18 | 27381.16 | SL hit (close<static) qty=1.00 sl=27301.70 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 27280.15 | 27511.62 | 27528.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 27135.70 | 27321.59 | 27387.80 | Break + close below crossover candle low |

### Cycle 10 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 27930.00 | 27443.27 | 27437.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 28098.95 | 27798.85 | 27666.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 27870.00 | 27894.47 | 27780.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 27870.00 | 27894.47 | 27780.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 27927.65 | 27984.09 | 27865.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 27927.65 | 27984.09 | 27865.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 27980.90 | 27983.45 | 27875.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 27980.90 | 27983.45 | 27875.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 28036.60 | 28174.01 | 28049.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 28036.60 | 28174.01 | 28049.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 27824.50 | 28104.11 | 28028.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 27824.50 | 28104.11 | 28028.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 27803.95 | 28044.08 | 28008.36 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 27736.35 | 27982.53 | 27983.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 15:15:00 | 27626.00 | 27911.22 | 27951.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 27470.40 | 27443.11 | 27578.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:15:00 | 27523.00 | 27443.11 | 27578.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 27492.75 | 27456.94 | 27542.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:15:00 | 27449.90 | 27455.55 | 27534.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 14:45:00 | 27445.95 | 27471.97 | 27528.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 27416.80 | 27471.97 | 27528.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 09:30:00 | 27352.60 | 27276.36 | 27354.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 27401.85 | 27301.46 | 27358.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-09 13:15:00 | 27640.00 | 27421.18 | 27403.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 27640.00 | 27421.18 | 27403.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 27781.00 | 27493.15 | 27437.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 10:15:00 | 27687.95 | 27726.05 | 27628.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 27687.95 | 27726.05 | 27628.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 27687.95 | 27726.05 | 27628.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 27654.45 | 27726.05 | 27628.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 27679.95 | 27716.83 | 27632.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 27679.95 | 27716.83 | 27632.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 27675.05 | 27708.48 | 27636.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 27686.30 | 27708.48 | 27636.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 27575.45 | 27704.32 | 27659.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 27575.45 | 27704.32 | 27659.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 27538.00 | 27671.05 | 27648.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:45:00 | 27537.00 | 27671.05 | 27648.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 27567.20 | 27629.04 | 27632.19 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 27680.20 | 27634.63 | 27633.89 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 27133.00 | 27539.96 | 27591.34 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 28095.80 | 27614.35 | 27582.74 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 27676.00 | 27782.19 | 27787.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 27238.85 | 27638.21 | 27717.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 27633.40 | 27571.05 | 27651.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 27633.40 | 27571.05 | 27651.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 27633.40 | 27571.05 | 27651.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 27633.40 | 27571.05 | 27651.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 27662.30 | 27589.30 | 27652.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 27667.85 | 27589.30 | 27652.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 27840.00 | 27639.44 | 27669.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 27840.00 | 27639.44 | 27669.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 27920.45 | 27695.64 | 27692.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 15:15:00 | 27991.00 | 27754.71 | 27719.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 15:15:00 | 27950.05 | 28047.56 | 27921.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:15:00 | 27967.75 | 28047.56 | 27921.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 27836.35 | 28005.32 | 27913.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 27836.35 | 28005.32 | 27913.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 27966.95 | 27997.65 | 27918.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 27895.10 | 27997.65 | 27918.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 27745.00 | 27947.12 | 27902.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:00:00 | 27745.00 | 27947.12 | 27902.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 27419.90 | 27841.67 | 27858.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 13:15:00 | 27240.15 | 27539.28 | 27584.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 13:15:00 | 27300.05 | 27267.00 | 27392.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-30 14:00:00 | 27300.05 | 27267.00 | 27392.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 27350.00 | 27283.60 | 27388.70 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 12:15:00 | 27686.05 | 27480.92 | 27453.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 14:15:00 | 27749.95 | 27551.35 | 27491.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 27600.00 | 27681.85 | 27611.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 27600.00 | 27681.85 | 27611.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 27600.00 | 27681.85 | 27611.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 27600.00 | 27681.85 | 27611.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 27730.80 | 27691.64 | 27622.42 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 27413.65 | 27584.01 | 27587.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 27336.85 | 27534.58 | 27564.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 26940.00 | 26866.58 | 27094.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:45:00 | 26554.00 | 26718.27 | 26952.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 14:15:00 | 25226.30 | 25743.42 | 26247.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-09 13:15:00 | 23898.60 | 24250.32 | 24810.69 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 22 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 24508.00 | 24394.29 | 24383.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 24678.00 | 24451.03 | 24410.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 24702.95 | 24774.93 | 24663.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 24731.90 | 24774.93 | 24663.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 24704.40 | 24760.82 | 24666.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 24639.90 | 24760.82 | 24666.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 24835.65 | 24764.80 | 24702.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 24962.55 | 24797.43 | 24746.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 24715.00 | 24831.74 | 24838.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 24715.00 | 24831.74 | 24838.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 24701.00 | 24776.58 | 24809.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 10:15:00 | 24882.80 | 24796.10 | 24812.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 10:15:00 | 24882.80 | 24796.10 | 24812.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 24882.80 | 24796.10 | 24812.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 24860.00 | 24796.10 | 24812.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 24885.00 | 24813.88 | 24818.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:45:00 | 24875.30 | 24813.88 | 24818.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 24875.15 | 24826.30 | 24823.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 24925.55 | 24846.15 | 24832.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 24842.50 | 24851.36 | 24837.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 24842.50 | 24851.36 | 24837.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 24842.50 | 24851.36 | 24837.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 24827.10 | 24851.36 | 24837.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 24830.90 | 24847.27 | 24837.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 24804.00 | 24847.27 | 24837.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 24810.00 | 24839.82 | 24834.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 24810.00 | 24839.82 | 24834.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 24755.05 | 24822.86 | 24827.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 24741.10 | 24803.63 | 24814.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 24810.05 | 24698.54 | 24734.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 24810.05 | 24698.54 | 24734.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 24810.05 | 24698.54 | 24734.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 24810.05 | 24698.54 | 24734.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 24833.00 | 24725.43 | 24743.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 24949.95 | 24725.43 | 24743.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 25382.35 | 24878.99 | 24811.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 25400.00 | 25061.54 | 24910.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 25336.25 | 25420.39 | 25228.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:00:00 | 25336.25 | 25420.39 | 25228.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 25589.90 | 25522.53 | 25406.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 25779.40 | 25638.37 | 25511.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 15:15:00 | 25521.65 | 25625.72 | 25633.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 25521.65 | 25625.72 | 25633.08 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 25796.50 | 25661.37 | 25645.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 25906.60 | 25754.21 | 25695.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 14:15:00 | 25741.80 | 25801.13 | 25741.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 25741.80 | 25801.13 | 25741.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 25741.80 | 25801.13 | 25741.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 25741.80 | 25801.13 | 25741.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 25701.00 | 25781.10 | 25738.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 25724.75 | 25781.10 | 25738.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 25746.15 | 25774.11 | 25738.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 12:30:00 | 25849.90 | 25760.81 | 25740.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 25594.05 | 25717.73 | 25723.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 25594.05 | 25717.73 | 25723.64 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 25971.00 | 25761.62 | 25739.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 26010.30 | 25871.25 | 25800.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 25830.05 | 25931.49 | 25872.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 13:15:00 | 25830.05 | 25931.49 | 25872.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 25830.05 | 25931.49 | 25872.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 25830.05 | 25931.49 | 25872.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 25872.75 | 25919.74 | 25872.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 25985.50 | 25905.79 | 25870.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 10:15:00 | 25788.40 | 25897.32 | 25873.46 | SL hit (close<static) qty=1.00 sl=25814.70 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 25665.10 | 25850.88 | 25854.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 12:15:00 | 25555.00 | 25791.70 | 25827.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 25031.55 | 24975.83 | 25135.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:30:00 | 24999.75 | 24975.83 | 25135.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 25081.80 | 24997.03 | 25130.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 25109.05 | 24997.03 | 25130.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 25125.65 | 25043.85 | 25120.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 25125.65 | 25043.85 | 25120.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 25137.20 | 25062.52 | 25122.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 25270.55 | 25124.12 | 25144.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 25420.30 | 25183.35 | 25169.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 25679.90 | 25328.82 | 25241.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 10:15:00 | 25882.85 | 25883.01 | 25699.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:00:00 | 25882.85 | 25883.01 | 25699.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 25734.55 | 25852.74 | 25731.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 25734.55 | 25852.74 | 25731.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 25945.00 | 25871.19 | 25750.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 25974.90 | 25889.09 | 25797.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 26165.20 | 26264.96 | 26274.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 11:15:00 | 26165.20 | 26264.96 | 26274.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 26075.15 | 26227.00 | 26256.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 25589.85 | 25461.24 | 25666.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 25589.85 | 25461.24 | 25666.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 24708.90 | 24539.01 | 24625.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 24711.00 | 24539.01 | 24625.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 24760.05 | 24583.22 | 24637.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 24782.85 | 24583.22 | 24637.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 24771.05 | 24681.91 | 24675.19 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 24485.85 | 24654.87 | 24665.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 24433.85 | 24610.66 | 24644.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 10:15:00 | 24479.90 | 24442.28 | 24520.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:45:00 | 24432.70 | 24442.28 | 24520.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 24279.15 | 24266.97 | 24352.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 24329.10 | 24266.97 | 24352.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 24322.30 | 24278.04 | 24350.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:30:00 | 24320.05 | 24278.04 | 24350.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 24300.05 | 24282.44 | 24345.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 24300.05 | 24282.44 | 24345.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 24349.55 | 24295.86 | 24345.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 24190.00 | 24295.86 | 24345.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 24277.95 | 24241.85 | 24293.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 24241.40 | 24241.85 | 24293.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 24282.25 | 24245.95 | 24290.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 24207.60 | 24238.28 | 24283.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 24612.60 | 24238.28 | 24283.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 24472.30 | 24285.08 | 24300.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 24472.30 | 24285.08 | 24300.33 | SL hit (close>static) qty=1.00 sl=24364.95 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 10:15:00 | 24500.00 | 24328.07 | 24318.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-22 15:15:00 | 24600.00 | 24426.75 | 24373.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 24901.25 | 25017.65 | 24839.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 10:00:00 | 24901.25 | 25017.65 | 24839.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 24887.40 | 24991.60 | 24844.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 24873.40 | 24991.60 | 24844.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 25016.20 | 24996.52 | 24859.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:30:00 | 25014.50 | 24996.52 | 24859.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 24849.25 | 24976.42 | 24874.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 24849.25 | 24976.42 | 24874.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 24875.05 | 24956.15 | 24874.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:45:00 | 24830.00 | 24956.15 | 24874.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 25020.00 | 24968.92 | 24888.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 25073.00 | 24968.92 | 24888.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 25115.90 | 24986.14 | 24903.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 10:15:00 | 25078.00 | 24986.14 | 24903.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 11:00:00 | 25065.65 | 25002.04 | 24918.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 25086.15 | 25072.60 | 25000.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 25107.35 | 25072.60 | 25000.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 25031.00 | 25064.28 | 25002.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 25013.35 | 25064.28 | 25002.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 25095.00 | 25070.42 | 25011.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 14:00:00 | 25147.60 | 25083.93 | 25027.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 15:15:00 | 25055.00 | 25209.93 | 25213.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 25055.00 | 25209.93 | 25213.27 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 25250.00 | 25217.94 | 25216.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 25351.00 | 25244.55 | 25228.82 | Break + close above crossover candle high |

### Cycle 39 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 25088.95 | 25213.43 | 25216.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 25019.25 | 25174.60 | 25198.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 24857.85 | 24791.69 | 24910.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 10:00:00 | 24857.85 | 24791.69 | 24910.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 24926.00 | 24811.88 | 24898.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 24926.00 | 24811.88 | 24898.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 25000.00 | 24849.51 | 24907.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 25000.00 | 24849.51 | 24907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 25025.05 | 24884.61 | 24918.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 25015.00 | 24884.61 | 24918.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 24962.25 | 24882.33 | 24903.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 24962.25 | 24882.33 | 24903.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 24799.30 | 24865.73 | 24893.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:15:00 | 24758.55 | 24865.73 | 24893.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 15:15:00 | 24758.85 | 24849.14 | 24883.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:15:00 | 23520.62 | 24365.39 | 24493.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:15:00 | 23520.91 | 24365.39 | 24493.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-12 14:15:00 | 24379.35 | 24245.10 | 24371.15 | SL hit (close>ema200) qty=0.50 sl=24245.10 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 15:15:00 | 24200.00 | 24133.16 | 24128.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 09:15:00 | 24255.30 | 24157.58 | 24139.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 13:15:00 | 24220.30 | 24231.30 | 24185.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 14:00:00 | 24220.30 | 24231.30 | 24185.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 24046.75 | 24194.39 | 24173.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 24046.75 | 24194.39 | 24173.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 24099.00 | 24175.31 | 24166.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 24327.00 | 24175.31 | 24166.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 10:15:00 | 26759.70 | 26198.49 | 25853.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 14:15:00 | 26608.85 | 26912.26 | 26914.42 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 27074.95 | 26919.49 | 26908.50 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 26591.10 | 26886.08 | 26899.42 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 27000.00 | 26806.50 | 26802.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 27183.15 | 26881.83 | 26837.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 14:15:00 | 27236.90 | 27259.09 | 27160.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 14:45:00 | 27232.15 | 27259.09 | 27160.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 27225.00 | 27252.27 | 27166.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 27247.75 | 27252.27 | 27166.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 10:45:00 | 27300.00 | 27262.00 | 27185.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 27482.00 | 27918.60 | 27924.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 27482.00 | 27918.60 | 27924.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 10:15:00 | 27425.30 | 27632.94 | 27745.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 27355.35 | 27279.43 | 27472.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:45:00 | 27342.70 | 27279.43 | 27472.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 27037.10 | 27189.97 | 27328.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 27225.10 | 27189.97 | 27328.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 25686.00 | 25967.60 | 26208.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:30:00 | 26151.35 | 25967.60 | 26208.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 25821.25 | 25589.45 | 25729.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 25821.25 | 25589.45 | 25729.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 26011.50 | 25673.86 | 25755.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:45:00 | 25879.40 | 25673.86 | 25755.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 26361.45 | 25811.38 | 25810.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 26630.25 | 25975.15 | 25884.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 26380.60 | 26417.13 | 26173.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 26380.60 | 26417.13 | 26173.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 26130.80 | 26320.88 | 26186.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 26130.80 | 26320.88 | 26186.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 26140.00 | 26284.71 | 26182.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 26140.00 | 26284.71 | 26182.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 26187.20 | 26251.65 | 26184.47 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 26010.00 | 26137.70 | 26146.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 25851.55 | 26080.47 | 26119.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 26106.05 | 26072.71 | 26108.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 26106.05 | 26072.71 | 26108.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 26106.05 | 26072.71 | 26108.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 26240.35 | 26072.71 | 26108.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 26194.70 | 26097.11 | 26116.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 26206.80 | 26097.11 | 26116.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 26186.60 | 26115.01 | 26122.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:15:00 | 25902.35 | 26115.01 | 26122.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:45:00 | 26116.40 | 26053.21 | 26076.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 25504.30 | 25339.17 | 25319.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 25504.30 | 25339.17 | 25319.38 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 25029.40 | 25334.22 | 25354.02 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 11:15:00 | 25374.55 | 25257.11 | 25255.13 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 25183.55 | 25242.40 | 25248.63 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 15:15:00 | 25312.15 | 25257.82 | 25253.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 25385.35 | 25283.33 | 25265.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 25776.55 | 25820.74 | 25646.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:30:00 | 25769.75 | 25820.74 | 25646.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 25748.00 | 25806.19 | 25655.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 25232.90 | 25806.19 | 25655.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 25337.40 | 25712.44 | 25626.58 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 25394.70 | 25547.96 | 25563.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 25231.85 | 25467.55 | 25523.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 25480.60 | 25420.99 | 25482.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 10:15:00 | 25480.60 | 25420.99 | 25482.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 25480.60 | 25420.99 | 25482.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 25480.60 | 25420.99 | 25482.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 25675.00 | 25471.79 | 25499.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 25675.00 | 25471.79 | 25499.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 26104.95 | 25598.42 | 25554.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 26399.05 | 25943.00 | 25748.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 26691.05 | 26743.21 | 26435.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 26634.20 | 26743.21 | 26435.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 26799.60 | 27417.80 | 27165.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 26552.85 | 27417.80 | 27165.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 27145.10 | 27363.26 | 27163.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 27300.00 | 27363.26 | 27163.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 27427.00 | 27312.37 | 27176.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 27250.00 | 27293.64 | 27191.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:30:00 | 27295.05 | 27207.21 | 27169.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 28031.95 | 28209.60 | 27945.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 28031.95 | 28209.60 | 27945.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 27777.70 | 28091.34 | 27935.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 27777.70 | 28091.34 | 27935.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 27787.40 | 28030.55 | 27922.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 27787.40 | 28030.55 | 27922.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 28419.40 | 28021.29 | 27940.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 15:15:00 | 27801.60 | 28041.43 | 28067.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 15:15:00 | 27801.60 | 28041.43 | 28067.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 27512.10 | 27935.56 | 28017.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 28003.55 | 27913.52 | 27990.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 28003.55 | 27913.52 | 27990.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 28003.55 | 27913.52 | 27990.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 28003.55 | 27913.52 | 27990.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 28222.00 | 27975.21 | 28011.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 28222.00 | 27975.21 | 28011.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 28065.95 | 27993.36 | 28016.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 28242.55 | 27993.36 | 28016.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 28281.05 | 28050.90 | 28040.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 15:15:00 | 28300.30 | 28100.78 | 28064.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 28314.65 | 28347.58 | 28222.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 28314.65 | 28347.58 | 28222.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 28227.80 | 28323.63 | 28223.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:30:00 | 28220.55 | 28323.63 | 28223.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 28178.20 | 28294.54 | 28219.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 28357.10 | 28294.54 | 28219.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 28049.95 | 28241.69 | 28207.98 | SL hit (close<static) qty=1.00 sl=28140.55 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 28263.50 | 28448.24 | 28456.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 28194.30 | 28331.84 | 28373.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 11:15:00 | 28295.40 | 28165.64 | 28242.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 11:15:00 | 28295.40 | 28165.64 | 28242.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 28295.40 | 28165.64 | 28242.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 28295.40 | 28165.64 | 28242.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 28253.45 | 28183.21 | 28243.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 27531.55 | 28218.70 | 28246.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 12:15:00 | 27969.95 | 27608.00 | 27596.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 27969.95 | 27608.00 | 27596.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 28176.35 | 27763.61 | 27671.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 28073.35 | 28110.33 | 27921.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:00:00 | 28073.35 | 28110.33 | 27921.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 28430.00 | 28147.84 | 27983.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 28470.00 | 28147.84 | 27983.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 15:15:00 | 27905.55 | 28061.99 | 28012.38 | SL hit (close<static) qty=1.00 sl=27950.45 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 27899.00 | 27994.78 | 28003.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 27778.00 | 27929.71 | 27964.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 11:15:00 | 27582.85 | 27581.82 | 27706.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 12:00:00 | 27582.85 | 27581.82 | 27706.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 27950.65 | 27619.83 | 27670.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 27993.00 | 27619.83 | 27670.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 27775.00 | 27650.87 | 27680.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:15:00 | 27730.00 | 27650.87 | 27680.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 12:15:00 | 27787.00 | 27701.95 | 27699.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 27787.00 | 27701.95 | 27699.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 27907.80 | 27757.29 | 27728.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 10:15:00 | 27648.30 | 27735.49 | 27720.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 10:15:00 | 27648.30 | 27735.49 | 27720.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 27648.30 | 27735.49 | 27720.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 27648.30 | 27735.49 | 27720.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 27733.55 | 27735.11 | 27722.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 13:00:00 | 27791.30 | 27746.34 | 27728.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-28 09:15:00 | 30570.43 | 30366.32 | 30045.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 12:15:00 | 30200.05 | 30309.44 | 30317.48 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 30579.45 | 30340.93 | 30327.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 30708.80 | 30451.79 | 30382.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 30682.30 | 30701.43 | 30548.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 30682.30 | 30701.43 | 30548.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 30660.95 | 30717.45 | 30616.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 30552.05 | 30717.45 | 30616.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 29938.00 | 30572.82 | 30568.70 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 30132.50 | 30484.76 | 30529.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 29634.75 | 30314.76 | 30447.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 30048.85 | 30016.34 | 30234.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 30450.00 | 30016.34 | 30234.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 30254.20 | 30063.91 | 30236.66 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 30550.00 | 30314.65 | 30303.67 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 09:15:00 | 30008.00 | 30297.18 | 30316.18 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 30810.00 | 30358.18 | 30315.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 30970.00 | 30548.84 | 30413.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 30740.00 | 30825.69 | 30649.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:30:00 | 30715.00 | 30825.69 | 30649.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 30745.00 | 30809.55 | 30657.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 30655.00 | 30809.55 | 30657.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 30800.00 | 30807.64 | 30670.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:30:00 | 30725.00 | 30807.64 | 30670.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 30595.00 | 30775.89 | 30680.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 30755.00 | 30775.89 | 30680.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 30600.00 | 30740.71 | 30673.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 30600.00 | 30740.71 | 30673.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 30770.00 | 30746.57 | 30682.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 30875.00 | 30808.26 | 30716.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 30965.00 | 31003.47 | 30871.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 30850.00 | 30952.16 | 30953.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 30850.00 | 30952.16 | 30953.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 11:15:00 | 30690.00 | 30899.73 | 30929.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 12:15:00 | 30695.00 | 30581.90 | 30702.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 12:15:00 | 30695.00 | 30581.90 | 30702.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 30695.00 | 30581.90 | 30702.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:45:00 | 30460.00 | 30573.54 | 30665.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:00:00 | 30425.00 | 30392.50 | 30533.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 30460.00 | 30418.00 | 30532.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:30:00 | 30425.00 | 30475.72 | 30530.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 30360.00 | 30457.65 | 30508.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:15:00 | 30550.00 | 30457.65 | 30508.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 30550.00 | 30476.12 | 30512.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 30440.00 | 30476.12 | 30512.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 30215.00 | 30423.89 | 30485.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 30010.00 | 30423.89 | 30485.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:00:00 | 30120.00 | 30083.24 | 30211.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 12:15:00 | 29660.00 | 29550.74 | 29548.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 29660.00 | 29550.74 | 29548.49 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 29100.00 | 29463.42 | 29510.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 28910.00 | 29352.74 | 29455.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 29330.00 | 29255.75 | 29388.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 11:15:00 | 29330.00 | 29255.75 | 29388.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 29330.00 | 29255.75 | 29388.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:30:00 | 29485.00 | 29255.75 | 29388.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 29255.00 | 29255.60 | 29375.91 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 29910.00 | 29457.01 | 29428.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 30040.00 | 29647.68 | 29524.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 29755.00 | 30026.84 | 29828.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 29755.00 | 30026.84 | 29828.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 29755.00 | 30026.84 | 29828.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 29755.00 | 30026.84 | 29828.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 30080.00 | 30037.47 | 29851.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:30:00 | 30130.00 | 30049.98 | 29874.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 30580.00 | 30041.98 | 29886.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 31415.00 | 31495.47 | 31500.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 31415.00 | 31495.47 | 31500.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 14:15:00 | 31400.00 | 31476.38 | 31490.95 | Break + close below crossover candle low |

### Cycle 72 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 31720.00 | 31508.08 | 31501.77 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 31265.00 | 31473.59 | 31497.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 31130.00 | 31404.87 | 31463.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 15:15:00 | 31100.00 | 31027.78 | 31156.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:15:00 | 30845.00 | 31027.78 | 31156.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 30540.00 | 30930.23 | 31100.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 30485.00 | 30930.23 | 31100.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 29625.00 | 29488.21 | 29481.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 29625.00 | 29488.21 | 29481.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 29745.00 | 29647.17 | 29587.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 29805.00 | 29915.82 | 29798.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 29805.00 | 29915.82 | 29798.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 29805.00 | 29915.82 | 29798.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 29805.00 | 29915.82 | 29798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 29985.00 | 29929.66 | 29815.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 30050.00 | 29929.66 | 29815.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 30025.00 | 29987.42 | 29875.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 29655.00 | 29871.16 | 29853.33 | SL hit (close<static) qty=1.00 sl=29755.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 29800.00 | 29839.42 | 29844.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 29670.00 | 29805.54 | 29828.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 29620.00 | 29572.06 | 29671.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 29600.00 | 29572.06 | 29671.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 29625.00 | 29587.12 | 29661.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 29680.00 | 29587.12 | 29661.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 29625.00 | 29594.69 | 29658.19 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 29895.00 | 29689.72 | 29685.48 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 29620.00 | 29697.53 | 29702.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 29570.00 | 29659.62 | 29683.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 11:15:00 | 29395.00 | 29384.82 | 29483.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:45:00 | 29365.00 | 29384.82 | 29483.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 29360.00 | 29368.66 | 29436.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 29180.00 | 29371.14 | 29426.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 29190.00 | 29336.91 | 29406.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:30:00 | 29180.00 | 29307.53 | 29386.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 14:15:00 | 29050.00 | 28820.07 | 28811.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 29050.00 | 28820.07 | 28811.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 29170.00 | 28890.06 | 28843.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 30720.00 | 30754.45 | 30305.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 30720.00 | 30754.45 | 30305.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 31340.00 | 31502.66 | 31357.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 31350.00 | 31475.13 | 31358.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 31325.00 | 31445.10 | 31355.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 31325.00 | 31445.10 | 31355.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 31230.00 | 31402.08 | 31344.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 31220.00 | 31402.08 | 31344.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 31370.00 | 31395.67 | 31346.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 31410.00 | 31395.67 | 31346.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 31425.00 | 31395.53 | 31350.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 31085.00 | 31307.19 | 31319.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 31085.00 | 31307.19 | 31319.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 31050.00 | 31255.75 | 31295.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 31170.00 | 31149.20 | 31215.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 31110.00 | 31149.20 | 31215.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 31025.00 | 31124.36 | 31197.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 30970.00 | 31109.49 | 31184.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 31290.00 | 31190.22 | 31206.77 | SL hit (close>static) qty=1.00 sl=31270.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 31300.00 | 31228.94 | 31222.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 31425.00 | 31268.15 | 31240.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 31190.00 | 31457.51 | 31381.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 31190.00 | 31457.51 | 31381.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 31190.00 | 31457.51 | 31381.56 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 31180.00 | 31325.19 | 31337.79 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 31405.00 | 31343.15 | 31337.33 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 31125.00 | 31297.42 | 31317.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 31000.00 | 31164.34 | 31231.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 31105.00 | 31031.20 | 31126.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 31105.00 | 31031.20 | 31126.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 31080.00 | 31040.96 | 31122.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 30960.00 | 31040.96 | 31122.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 30840.00 | 31000.77 | 31096.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 30800.00 | 30959.62 | 31069.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 30780.00 | 30877.42 | 30968.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 30775.00 | 30856.94 | 30951.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 30830.00 | 30839.05 | 30910.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 30920.00 | 30856.99 | 30906.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 30760.00 | 30856.99 | 30906.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:00:00 | 30800.00 | 30845.02 | 30888.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 30790.00 | 30831.01 | 30874.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 31050.00 | 30908.28 | 30900.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 31050.00 | 30908.28 | 30900.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 31370.00 | 31068.64 | 30981.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 31755.00 | 32007.15 | 31666.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 31755.00 | 32007.15 | 31666.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 31755.00 | 32007.15 | 31666.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 31755.00 | 32007.15 | 31666.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 31725.00 | 31950.72 | 31672.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 31725.00 | 31950.72 | 31672.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 31725.00 | 31905.58 | 31676.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 31795.00 | 31879.46 | 31685.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:45:00 | 31815.00 | 31887.57 | 31707.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 31830.00 | 31920.80 | 31830.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 31495.00 | 31773.01 | 31780.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 31495.00 | 31773.01 | 31780.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 30955.00 | 31609.41 | 31705.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 31370.00 | 31216.95 | 31420.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 31370.00 | 31216.95 | 31420.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 31370.00 | 31216.95 | 31420.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 31370.00 | 31216.95 | 31420.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 31270.00 | 31227.56 | 31407.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 31160.00 | 31214.05 | 31384.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 30960.00 | 30747.14 | 30746.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 30960.00 | 30747.14 | 30746.16 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 30660.00 | 30736.57 | 30741.95 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 30885.00 | 30766.25 | 30754.96 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 30665.00 | 30746.00 | 30746.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 30615.00 | 30719.80 | 30734.80 | Break + close below crossover candle low |

### Cycle 90 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 30875.00 | 30750.84 | 30747.55 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 30700.00 | 30740.67 | 30743.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 30535.00 | 30673.42 | 30709.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 30600.00 | 30599.08 | 30644.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 30600.00 | 30599.08 | 30644.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 30600.00 | 30599.08 | 30644.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 30155.00 | 30397.79 | 30526.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:45:00 | 30155.00 | 30286.09 | 30426.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 30200.00 | 30265.88 | 30404.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 30690.00 | 30429.84 | 30429.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 30690.00 | 30429.84 | 30429.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 10:15:00 | 30715.00 | 30550.92 | 30490.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 11:15:00 | 30500.00 | 30540.73 | 30490.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 30500.00 | 30540.73 | 30490.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 30500.00 | 30540.73 | 30490.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 30500.00 | 30540.73 | 30490.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 30465.00 | 30525.59 | 30488.63 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 30375.00 | 30470.18 | 30470.48 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 30495.00 | 30475.14 | 30472.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 30600.00 | 30500.12 | 30484.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 15:15:00 | 30455.00 | 30596.05 | 30551.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 15:15:00 | 30455.00 | 30596.05 | 30551.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 30455.00 | 30596.05 | 30551.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 30550.00 | 30596.05 | 30551.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 30530.00 | 30582.84 | 30549.89 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 30430.00 | 30540.20 | 30540.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 30310.00 | 30414.88 | 30458.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 30505.00 | 30401.56 | 30434.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 30505.00 | 30401.56 | 30434.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 30505.00 | 30401.56 | 30434.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 30505.00 | 30401.56 | 30434.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 30450.00 | 30411.25 | 30435.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 31455.00 | 30411.25 | 30435.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 31710.00 | 30671.00 | 30551.54 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 30685.00 | 30887.22 | 30913.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 30630.00 | 30835.78 | 30888.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 30200.00 | 30198.86 | 30404.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:30:00 | 30340.00 | 30198.86 | 30404.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 30265.00 | 30117.37 | 30236.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 30265.00 | 30117.37 | 30236.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 30470.00 | 30187.90 | 30257.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 30470.00 | 30187.90 | 30257.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 29635.00 | 29556.80 | 29691.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 29685.00 | 29556.80 | 29691.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 29770.00 | 29599.44 | 29699.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 29770.00 | 29599.44 | 29699.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 29825.00 | 29644.55 | 29710.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 29930.00 | 29644.55 | 29710.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 30080.00 | 29797.37 | 29769.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 30140.00 | 29903.12 | 29824.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 30020.00 | 30062.78 | 29976.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 30020.00 | 30062.78 | 29976.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 30020.00 | 30062.78 | 29976.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 29925.00 | 30062.78 | 29976.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 29960.00 | 30042.22 | 29974.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 30095.00 | 30042.22 | 29974.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 30055.00 | 30038.78 | 29979.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 30045.00 | 30019.21 | 29986.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 13:15:00 | 29855.00 | 29986.37 | 29974.88 | SL hit (close<static) qty=1.00 sl=29880.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 29885.00 | 29962.18 | 29965.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 29710.00 | 29885.80 | 29928.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 29945.00 | 29887.91 | 29921.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 29945.00 | 29887.91 | 29921.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 29945.00 | 29887.91 | 29921.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 29945.00 | 29887.91 | 29921.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 30020.00 | 29914.33 | 29930.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 30020.00 | 29914.33 | 29930.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 29910.00 | 29913.46 | 29928.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 30170.00 | 29913.46 | 29928.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 30270.00 | 29984.77 | 29959.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 30350.00 | 30089.85 | 30013.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 30360.00 | 30369.85 | 30232.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 30360.00 | 30369.85 | 30232.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 30240.00 | 30339.10 | 30241.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 30240.00 | 30339.10 | 30241.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 30200.00 | 30311.28 | 30237.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 30165.00 | 30311.28 | 30237.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 30195.00 | 30288.02 | 30233.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 30375.00 | 30288.02 | 30233.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 30075.00 | 30245.42 | 30219.52 | SL hit (close<static) qty=1.00 sl=30115.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 29955.00 | 30187.34 | 30195.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 29900.00 | 30129.87 | 30168.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 29785.00 | 29775.72 | 29867.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:45:00 | 29800.00 | 29775.72 | 29867.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 29345.00 | 29621.87 | 29760.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 29295.00 | 29621.87 | 29760.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:45:00 | 29280.00 | 29439.26 | 29606.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 29880.00 | 29609.39 | 29604.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 29880.00 | 29609.39 | 29604.29 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 29780.00 | 29813.68 | 29815.98 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 30040.00 | 29858.94 | 29836.35 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 29765.00 | 29832.22 | 29841.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 29470.00 | 29736.80 | 29774.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 29220.00 | 29073.58 | 29278.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 29220.00 | 29073.58 | 29278.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 29325.00 | 29123.86 | 29282.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 29275.00 | 29123.86 | 29282.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 29900.00 | 29279.09 | 29338.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 29900.00 | 29279.09 | 29338.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 29645.00 | 29352.27 | 29366.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 29405.00 | 29366.82 | 29371.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 29530.00 | 29399.45 | 29386.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 29530.00 | 29399.45 | 29386.30 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 29285.00 | 29376.56 | 29377.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 29190.00 | 29326.20 | 29353.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 29440.00 | 29335.97 | 29352.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 15:15:00 | 29440.00 | 29335.97 | 29352.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 29440.00 | 29335.97 | 29352.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 29415.00 | 29335.97 | 29352.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 29085.00 | 29285.77 | 29327.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 29330.00 | 29285.77 | 29327.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 29130.00 | 29175.98 | 29253.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 29130.00 | 29175.98 | 29253.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 29270.00 | 29194.79 | 29255.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 29270.00 | 29194.79 | 29255.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 29400.00 | 29235.83 | 29268.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 29065.00 | 29235.83 | 29268.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 29195.00 | 29134.63 | 29172.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 29315.00 | 29218.97 | 29206.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 29315.00 | 29218.97 | 29206.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 29525.00 | 29280.18 | 29235.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 29440.00 | 29465.75 | 29372.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 29430.00 | 29465.75 | 29372.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 29230.00 | 29418.60 | 29359.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 29230.00 | 29418.60 | 29359.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 29320.00 | 29398.88 | 29356.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 29240.00 | 29398.88 | 29356.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 29290.00 | 29367.69 | 29348.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 29290.00 | 29367.69 | 29348.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 29295.00 | 29353.15 | 29343.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:15:00 | 29350.00 | 29353.15 | 29343.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 29350.00 | 29352.52 | 29344.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 29400.00 | 29352.52 | 29344.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 29355.00 | 29460.20 | 29460.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 29350.00 | 29444.53 | 29453.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 29350.00 | 29444.53 | 29453.43 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 29640.00 | 29482.10 | 29468.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 29745.00 | 29585.96 | 29541.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 30025.00 | 30047.23 | 29886.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 30025.00 | 30047.23 | 29886.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 29625.00 | 29948.83 | 29868.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 29600.00 | 29948.83 | 29868.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 29850.00 | 29929.06 | 29867.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 29645.00 | 29929.06 | 29867.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 29925.00 | 29928.25 | 29872.45 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 29650.00 | 29808.26 | 29828.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 29210.00 | 29688.61 | 29772.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 29155.00 | 29063.43 | 29298.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 29155.00 | 29063.43 | 29298.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 29155.00 | 29063.43 | 29298.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 29160.00 | 29063.43 | 29298.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 29080.00 | 29083.80 | 29267.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 28965.00 | 29083.80 | 29267.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 29030.00 | 28784.20 | 28756.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 29030.00 | 28784.20 | 28756.76 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 28760.00 | 28832.90 | 28835.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 28650.00 | 28778.25 | 28809.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 27600.00 | 27531.60 | 27722.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 27600.00 | 27531.60 | 27722.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 27675.00 | 27560.28 | 27717.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 27675.00 | 27560.28 | 27717.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 27485.00 | 27545.22 | 27696.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:30:00 | 27675.00 | 27545.22 | 27696.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 26990.00 | 27070.24 | 27222.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 26925.00 | 27049.56 | 27111.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 11:15:00 | 26720.00 | 26571.03 | 26567.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 26720.00 | 26571.03 | 26567.17 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 26470.00 | 26557.89 | 26563.21 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 26655.00 | 26512.56 | 26510.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 26960.00 | 26726.57 | 26625.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 26665.00 | 26749.21 | 26655.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 26665.00 | 26749.21 | 26655.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 26665.00 | 26749.21 | 26655.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 26665.00 | 26749.21 | 26655.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 26705.00 | 26740.37 | 26660.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:30:00 | 26700.00 | 26740.37 | 26660.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 26700.00 | 26732.29 | 26663.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 26780.00 | 26729.73 | 26678.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 26610.00 | 26713.83 | 26680.99 | SL hit (close<static) qty=1.00 sl=26615.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 26440.00 | 26659.06 | 26659.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 26400.00 | 26568.20 | 26615.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 26435.00 | 26431.50 | 26496.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 26435.00 | 26431.50 | 26496.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 26435.00 | 26431.50 | 26496.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 26485.00 | 26431.50 | 26496.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 26550.00 | 26392.16 | 26432.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 26550.00 | 26392.16 | 26432.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 26645.00 | 26442.73 | 26451.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 26325.00 | 26442.73 | 26451.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 26365.00 | 26367.62 | 26405.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:15:00 | 26335.00 | 26367.62 | 26405.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 26320.00 | 26348.10 | 26393.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:45:00 | 26320.00 | 26341.48 | 26386.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:45:00 | 26320.00 | 26308.92 | 26357.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 26455.00 | 26325.24 | 26347.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 26455.00 | 26325.24 | 26347.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 26400.00 | 26340.20 | 26352.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 26300.00 | 26340.20 | 26352.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 26175.00 | 26032.92 | 26032.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 26175.00 | 26032.92 | 26032.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 26395.00 | 26281.01 | 26206.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 26200.00 | 26380.34 | 26307.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 26200.00 | 26380.34 | 26307.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 26200.00 | 26380.34 | 26307.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 26200.00 | 26380.34 | 26307.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 26065.00 | 26317.27 | 26285.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 26085.00 | 26317.27 | 26285.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 26035.00 | 26260.82 | 26262.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 26000.00 | 26208.66 | 26239.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 25615.00 | 25592.62 | 25756.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:45:00 | 25635.00 | 25592.62 | 25756.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 25650.00 | 25605.28 | 25734.29 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 10:15:00 | 26000.00 | 25785.41 | 25764.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 26080.00 | 25844.33 | 25792.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 26200.00 | 26312.41 | 26189.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 26200.00 | 26312.41 | 26189.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 26250.00 | 26299.93 | 26195.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 26250.00 | 26299.93 | 26195.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 26265.00 | 26292.94 | 26201.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:30:00 | 26255.00 | 26292.94 | 26201.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 26220.00 | 26278.35 | 26203.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 26280.00 | 26276.68 | 26209.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 26160.00 | 26246.30 | 26211.83 | SL hit (close<static) qty=1.00 sl=26170.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 27120.00 | 27342.88 | 27369.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 27000.00 | 27231.44 | 27311.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 26960.00 | 26949.53 | 27066.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:45:00 | 27035.00 | 26949.53 | 27066.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 26995.00 | 26904.01 | 26984.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 26995.00 | 26904.01 | 26984.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 27145.00 | 26952.21 | 26999.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 27145.00 | 26952.21 | 26999.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 27125.00 | 26986.76 | 27010.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 27190.00 | 26986.76 | 27010.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 27025.00 | 27006.13 | 27016.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 26980.00 | 27006.13 | 27016.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 27195.00 | 27043.90 | 27032.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 27300.00 | 27143.47 | 27092.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 11:15:00 | 27710.00 | 27723.20 | 27572.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 12:00:00 | 27710.00 | 27723.20 | 27572.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 27575.00 | 27683.45 | 27579.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 27575.00 | 27683.45 | 27579.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 27495.00 | 27645.76 | 27571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 27505.00 | 27645.76 | 27571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 27500.00 | 27616.61 | 27565.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 27550.00 | 27616.61 | 27565.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 27600.00 | 27663.22 | 27604.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 27600.00 | 27663.22 | 27604.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 27290.00 | 27588.58 | 27576.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 27290.00 | 27588.58 | 27576.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 27135.00 | 27497.86 | 27535.93 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 27465.00 | 27430.70 | 27426.39 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 27300.00 | 27404.56 | 27414.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 27145.00 | 27352.65 | 27390.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 27495.00 | 27263.44 | 27322.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 27495.00 | 27263.44 | 27322.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 27495.00 | 27263.44 | 27322.59 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 12:15:00 | 27510.00 | 27357.68 | 27355.18 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 27195.00 | 27325.52 | 27341.02 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 27455.00 | 27334.30 | 27321.21 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 27005.00 | 27268.44 | 27292.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 26620.00 | 26968.84 | 27052.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 26535.00 | 26522.44 | 26735.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 26535.00 | 26522.44 | 26735.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 26765.00 | 26596.96 | 26733.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 26765.00 | 26596.96 | 26733.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 26790.00 | 26635.57 | 26738.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 27110.00 | 26635.57 | 26738.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 27075.00 | 26723.45 | 26769.51 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 27025.00 | 26824.81 | 26810.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 27060.00 | 26895.26 | 26847.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 26920.00 | 26961.57 | 26896.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 26920.00 | 26961.57 | 26896.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 27105.00 | 27206.08 | 27133.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 27060.00 | 27206.08 | 27133.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 27195.00 | 27203.87 | 27139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 27085.00 | 27203.87 | 27139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 27205.00 | 27204.09 | 27145.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 27235.00 | 27204.09 | 27145.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 27260.00 | 27215.28 | 27155.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 27235.00 | 27215.28 | 27155.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 26820.00 | 27183.95 | 27165.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 26530.00 | 27183.95 | 27165.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 12:15:00 | 27125.00 | 27149.58 | 27152.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 09:15:00 | 26720.00 | 27044.44 | 27100.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 26730.00 | 26670.32 | 26815.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:30:00 | 26715.00 | 26670.32 | 26815.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 26825.00 | 26718.01 | 26812.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 26815.00 | 26718.01 | 26812.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 26790.00 | 26732.40 | 26810.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 26790.00 | 26732.40 | 26810.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 26800.00 | 26745.92 | 26809.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 26580.00 | 26745.92 | 26809.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 26365.00 | 26305.00 | 26302.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 26365.00 | 26305.00 | 26302.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 26525.00 | 26363.40 | 26331.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 12:15:00 | 26485.00 | 26688.13 | 26566.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 26485.00 | 26688.13 | 26566.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 26485.00 | 26688.13 | 26566.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 26485.00 | 26688.13 | 26566.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 26285.00 | 26607.51 | 26541.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 26285.00 | 26607.51 | 26541.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 26230.00 | 26466.80 | 26484.59 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 26610.00 | 26492.65 | 26485.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 26790.00 | 26559.67 | 26518.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 12:15:00 | 26435.00 | 26549.87 | 26525.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 26435.00 | 26549.87 | 26525.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 26435.00 | 26549.87 | 26525.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 26410.00 | 26549.87 | 26525.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 26590.00 | 26557.90 | 26531.77 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 26375.00 | 26496.00 | 26508.00 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 26605.00 | 26517.80 | 26516.82 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 26460.00 | 26506.24 | 26511.65 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 26595.00 | 26516.60 | 26514.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 26725.00 | 26567.41 | 26539.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 26740.00 | 26789.38 | 26701.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 26740.00 | 26789.38 | 26701.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 26705.00 | 26772.50 | 26701.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 26765.00 | 26772.50 | 26701.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 26660.00 | 26750.00 | 26698.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 26660.00 | 26750.00 | 26698.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 26745.00 | 26749.00 | 26702.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 26850.00 | 26749.00 | 26702.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 26380.00 | 26691.36 | 26685.18 | SL hit (close<static) qty=1.00 sl=26630.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 26370.00 | 26627.09 | 26656.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 26290.00 | 26559.67 | 26623.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 25970.00 | 25910.09 | 26189.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:00:00 | 25970.00 | 25910.09 | 26189.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 26155.00 | 25959.07 | 26186.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 26155.00 | 25959.07 | 26186.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 25805.00 | 25928.26 | 26152.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 25200.00 | 25928.26 | 26152.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 23940.00 | 24738.83 | 25055.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 24170.00 | 24036.30 | 24451.45 | SL hit (close>ema200) qty=0.50 sl=24036.30 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 23680.00 | 23447.23 | 23415.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 23820.00 | 23561.43 | 23475.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 23360.00 | 23722.75 | 23634.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 23360.00 | 23722.75 | 23634.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 23360.00 | 23722.75 | 23634.86 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 23310.00 | 23560.56 | 23571.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 23205.00 | 23489.45 | 23538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 23560.00 | 23438.27 | 23491.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 23560.00 | 23438.27 | 23491.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 23560.00 | 23438.27 | 23491.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 23595.00 | 23438.27 | 23491.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 23650.00 | 23480.62 | 23505.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 23650.00 | 23480.62 | 23505.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 23560.00 | 23496.49 | 23510.42 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 23615.00 | 23520.19 | 23519.92 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 23065.00 | 23441.82 | 23486.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 22840.00 | 23198.05 | 23353.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 22950.00 | 22913.21 | 23112.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 22950.00 | 22913.21 | 23112.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 23325.00 | 22995.57 | 23131.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 23325.00 | 22995.57 | 23131.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 23430.00 | 23082.45 | 23158.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 23380.00 | 23082.45 | 23158.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 24000.00 | 23342.22 | 23264.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 24135.00 | 23708.29 | 23469.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 23660.00 | 23895.59 | 23653.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 23660.00 | 23895.59 | 23653.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 23660.00 | 23895.59 | 23653.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 23695.00 | 23895.59 | 23653.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 23800.00 | 23876.47 | 23667.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 23850.00 | 23835.94 | 23683.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 23620.00 | 23811.00 | 23699.61 | SL hit (close<static) qty=1.00 sl=23650.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 23385.00 | 23643.59 | 23644.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 11:15:00 | 23300.00 | 23574.87 | 23613.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 23385.00 | 23325.69 | 23457.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 23385.00 | 23325.69 | 23457.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 23385.00 | 23325.69 | 23457.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 23380.00 | 23325.69 | 23457.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 23295.00 | 23306.23 | 23414.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:30:00 | 23405.00 | 23306.23 | 23414.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 23210.00 | 23087.37 | 23218.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:45:00 | 23165.00 | 23087.37 | 23218.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 23165.00 | 23102.90 | 23213.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:15:00 | 23120.00 | 23102.90 | 23213.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 23120.00 | 23106.32 | 23204.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 23190.00 | 23106.32 | 23204.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 23160.00 | 23117.06 | 23200.78 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 23685.00 | 23287.40 | 23251.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 24380.00 | 23535.67 | 23394.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 23860.00 | 24086.23 | 23827.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 23860.00 | 24086.23 | 23827.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 23860.00 | 24086.23 | 23827.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 23860.00 | 24086.23 | 23827.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 23935.00 | 24055.99 | 23837.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 24060.00 | 24035.79 | 23848.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 15:15:00 | 23755.00 | 23994.01 | 23891.09 | SL hit (close<static) qty=1.00 sl=23830.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 25325.00 | 25460.61 | 25463.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 24910.00 | 25248.02 | 25352.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 25130.00 | 25126.69 | 25252.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 25130.00 | 25126.69 | 25252.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 25130.00 | 25126.69 | 25252.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 25375.00 | 25126.69 | 25252.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 25160.00 | 25133.35 | 25244.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 25250.00 | 25133.35 | 25244.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 25325.00 | 25159.95 | 25236.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 25355.00 | 25159.95 | 25236.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 25365.00 | 25200.96 | 25248.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 25325.00 | 25200.96 | 25248.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 24930.00 | 25135.93 | 25205.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 24825.00 | 25073.74 | 25170.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 24840.00 | 24940.19 | 25066.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 24785.00 | 25051.94 | 25086.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 24830.00 | 24522.02 | 24602.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 24750.00 | 24623.03 | 24633.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 24730.00 | 24644.43 | 24642.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 24730.00 | 24644.43 | 24642.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 24870.00 | 24689.54 | 24663.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 24585.00 | 24699.91 | 24674.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 24585.00 | 24699.91 | 24674.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 24585.00 | 24699.91 | 24674.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 24585.00 | 24699.91 | 24674.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 24710.00 | 24701.93 | 24677.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:15:00 | 24750.00 | 24701.93 | 24677.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:00:00 | 24750.00 | 24711.54 | 24684.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 24890.00 | 24802.45 | 24759.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-31 10:15:00 | 25028.35 | 2024-06-03 09:15:00 | 25477.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-05-31 11:45:00 | 25002.60 | 2024-06-03 09:15:00 | 25477.05 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-05-31 12:30:00 | 25014.20 | 2024-06-03 09:15:00 | 25477.05 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-06-18 15:15:00 | 27548.95 | 2024-06-19 09:15:00 | 27194.05 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-06-19 11:15:00 | 27599.95 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-19 13:45:00 | 27526.60 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-19 14:15:00 | 27525.00 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-20 11:15:00 | 27720.00 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-20 12:15:00 | 27668.90 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-06-20 12:45:00 | 27709.45 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-06-20 14:00:00 | 27746.05 | 2024-06-21 14:15:00 | 27307.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-06-21 11:45:00 | 27675.00 | 2024-06-21 15:15:00 | 27280.15 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-06-21 13:15:00 | 27653.45 | 2024-06-21 15:15:00 | 27280.15 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-06-21 13:45:00 | 27678.00 | 2024-06-21 15:15:00 | 27280.15 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-05 13:15:00 | 27449.90 | 2024-07-09 13:15:00 | 27640.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-07-05 14:45:00 | 27445.95 | 2024-07-09 13:15:00 | 27640.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-07-05 15:15:00 | 27416.80 | 2024-07-09 13:15:00 | 27640.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-07-09 09:30:00 | 27352.60 | 2024-07-09 13:15:00 | 27640.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest1 | 2024-08-06 13:45:00 | 26554.00 | 2024-08-07 14:15:00 | 25226.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-06 13:45:00 | 26554.00 | 2024-08-09 13:15:00 | 23898.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-13 10:15:00 | 24180.65 | 2024-08-16 13:15:00 | 24508.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-13 12:45:00 | 24216.15 | 2024-08-16 13:15:00 | 24508.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-13 14:45:00 | 24216.65 | 2024-08-16 13:15:00 | 24508.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-14 09:15:00 | 24125.00 | 2024-08-16 13:15:00 | 24508.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-08-22 09:15:00 | 24962.55 | 2024-08-23 12:15:00 | 24715.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-04 15:00:00 | 25779.40 | 2024-09-06 15:15:00 | 25521.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-11 12:30:00 | 25849.90 | 2024-09-11 14:15:00 | 25594.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-16 09:15:00 | 25985.50 | 2024-09-16 10:15:00 | 25788.40 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-09-26 11:30:00 | 25974.90 | 2024-10-04 11:15:00 | 26165.20 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-10-21 09:15:00 | 24190.00 | 2024-10-22 09:15:00 | 24472.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-21 13:30:00 | 24277.95 | 2024-10-22 09:15:00 | 24472.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-21 14:00:00 | 24241.40 | 2024-10-22 09:15:00 | 24472.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-21 14:45:00 | 24282.25 | 2024-10-22 09:15:00 | 24472.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-10-28 09:15:00 | 25073.00 | 2024-10-31 15:15:00 | 25055.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-10-28 09:45:00 | 25115.90 | 2024-10-31 15:15:00 | 25055.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-10-28 10:15:00 | 25078.00 | 2024-10-31 15:15:00 | 25055.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-10-28 11:00:00 | 25065.65 | 2024-10-31 15:15:00 | 25055.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-10-29 14:00:00 | 25147.60 | 2024-10-31 15:15:00 | 25055.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-11-07 14:15:00 | 24758.55 | 2024-11-12 09:15:00 | 23520.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 15:15:00 | 24758.85 | 2024-11-12 09:15:00 | 23520.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 24758.55 | 2024-11-12 14:15:00 | 24379.35 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2024-11-07 15:15:00 | 24758.85 | 2024-11-12 14:15:00 | 24379.35 | STOP_HIT | 0.50 | 1.53% |
| BUY | retest2 | 2024-11-22 09:15:00 | 24327.00 | 2024-12-02 10:15:00 | 26759.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-13 09:15:00 | 27247.75 | 2024-12-19 09:15:00 | 27482.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-12-13 10:45:00 | 27300.00 | 2024-12-19 09:15:00 | 27482.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-01-07 12:15:00 | 25902.35 | 2025-01-16 10:15:00 | 25504.30 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-01-08 10:45:00 | 26116.40 | 2025-01-16 10:15:00 | 25504.30 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-02-01 14:15:00 | 27300.00 | 2025-02-11 15:15:00 | 27801.60 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2025-02-03 09:15:00 | 27427.00 | 2025-02-11 15:15:00 | 27801.60 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2025-02-03 10:45:00 | 27250.00 | 2025-02-11 15:15:00 | 27801.60 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-02-03 13:30:00 | 27295.05 | 2025-02-11 15:15:00 | 27801.60 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-02-14 09:15:00 | 28357.10 | 2025-02-14 10:15:00 | 28049.95 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-02-14 11:45:00 | 28299.95 | 2025-02-17 12:15:00 | 28131.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-02-14 12:15:00 | 28293.00 | 2025-02-17 12:15:00 | 28131.05 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-02-14 13:00:00 | 28356.65 | 2025-02-17 12:15:00 | 28131.05 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-02-19 09:15:00 | 28619.95 | 2025-02-21 10:15:00 | 28263.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-02-20 11:30:00 | 28516.40 | 2025-02-21 10:15:00 | 28263.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-02-20 12:15:00 | 28547.65 | 2025-02-21 10:15:00 | 28263.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-02-20 14:15:00 | 28507.75 | 2025-02-21 10:15:00 | 28263.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-02-28 09:15:00 | 27531.55 | 2025-03-05 12:15:00 | 27969.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-07 10:15:00 | 28470.00 | 2025-03-07 15:15:00 | 27905.55 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-03-17 11:15:00 | 27730.00 | 2025-03-17 12:15:00 | 27787.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-03-18 13:00:00 | 27791.30 | 2025-03-28 09:15:00 | 30570.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:45:00 | 30875.00 | 2025-04-23 10:15:00 | 30850.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-04-21 12:00:00 | 30965.00 | 2025-04-23 10:15:00 | 30850.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-04-25 09:45:00 | 30460.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2025-04-25 14:00:00 | 30425.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-04-25 15:15:00 | 30460.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2025-04-28 11:30:00 | 30425.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-04-29 10:15:00 | 30010.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-04-30 12:00:00 | 30120.00 | 2025-05-08 12:15:00 | 29660.00 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2025-05-13 14:30:00 | 30130.00 | 2025-05-22 13:15:00 | 31415.00 | STOP_HIT | 1.00 | 4.26% |
| BUY | retest2 | 2025-05-14 09:15:00 | 30580.00 | 2025-05-22 13:15:00 | 31415.00 | STOP_HIT | 1.00 | 2.73% |
| SELL | retest2 | 2025-05-28 10:15:00 | 30485.00 | 2025-06-06 09:15:00 | 29625.00 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-06-11 09:15:00 | 30050.00 | 2025-06-11 15:15:00 | 29655.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-06-11 12:15:00 | 30025.00 | 2025-06-11 15:15:00 | 29655.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-20 12:15:00 | 29180.00 | 2025-06-25 14:15:00 | 29050.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-06-20 12:45:00 | 29190.00 | 2025-06-25 14:15:00 | 29050.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-06-20 13:30:00 | 29180.00 | 2025-06-25 14:15:00 | 29050.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-07-04 13:15:00 | 31410.00 | 2025-07-07 09:15:00 | 31085.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-04 13:45:00 | 31425.00 | 2025-07-07 09:15:00 | 31085.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-08 10:30:00 | 30970.00 | 2025-07-08 13:15:00 | 31290.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-16 10:30:00 | 30800.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-07-17 10:15:00 | 30780.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-17 11:00:00 | 30775.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-17 14:30:00 | 30830.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-18 10:15:00 | 30760.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-18 13:00:00 | 30800.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-07-18 14:30:00 | 30790.00 | 2025-07-21 10:15:00 | 31050.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-23 13:15:00 | 31795.00 | 2025-07-25 09:15:00 | 31495.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-23 13:45:00 | 31815.00 | 2025-07-25 09:15:00 | 31495.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-24 14:15:00 | 31830.00 | 2025-07-25 09:15:00 | 31495.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-28 12:00:00 | 31160.00 | 2025-07-31 13:15:00 | 30960.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-08-05 14:00:00 | 30155.00 | 2025-08-07 14:15:00 | 30690.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-08-06 10:45:00 | 30155.00 | 2025-08-07 14:15:00 | 30690.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-08-06 11:30:00 | 30200.00 | 2025-08-07 14:15:00 | 30690.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-09-04 09:15:00 | 30095.00 | 2025-09-04 13:15:00 | 29855.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-04 10:15:00 | 30055.00 | 2025-09-04 13:15:00 | 29855.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-04 12:30:00 | 30045.00 | 2025-09-04 13:15:00 | 29855.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-09-10 09:15:00 | 30375.00 | 2025-09-10 09:15:00 | 30075.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-15 10:15:00 | 29295.00 | 2025-09-17 09:15:00 | 29880.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-15 14:45:00 | 29280.00 | 2025-09-17 09:15:00 | 29880.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-30 09:30:00 | 29405.00 | 2025-09-30 10:15:00 | 29530.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-03 09:15:00 | 29065.00 | 2025-10-06 15:15:00 | 29315.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-06 12:30:00 | 29195.00 | 2025-10-06 15:15:00 | 29315.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-10-09 09:15:00 | 29400.00 | 2025-10-13 11:15:00 | 29350.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-10-13 10:00:00 | 29355.00 | 2025-10-13 11:15:00 | 29350.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-10-23 12:15:00 | 28965.00 | 2025-10-29 11:15:00 | 29030.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-14 09:45:00 | 26925.00 | 2025-11-21 11:15:00 | 26720.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-11-27 15:15:00 | 26780.00 | 2025-11-28 09:15:00 | 26610.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-12-03 13:15:00 | 26335.00 | 2025-12-11 12:15:00 | 26175.00 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-12-03 13:45:00 | 26320.00 | 2025-12-11 12:15:00 | 26175.00 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-12-03 14:45:00 | 26320.00 | 2025-12-11 12:15:00 | 26175.00 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-12-04 10:45:00 | 26320.00 | 2025-12-11 12:15:00 | 26175.00 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-12-05 09:15:00 | 26300.00 | 2025-12-11 12:15:00 | 26175.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-12-29 15:15:00 | 26280.00 | 2025-12-30 10:15:00 | 26160.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-30 12:30:00 | 26315.00 | 2026-01-08 09:15:00 | 27120.00 | STOP_HIT | 1.00 | 3.06% |
| BUY | retest2 | 2025-12-30 15:00:00 | 26365.00 | 2026-01-08 09:15:00 | 27120.00 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2025-12-31 09:45:00 | 26340.00 | 2026-01-08 09:15:00 | 27120.00 | STOP_HIT | 1.00 | 2.96% |
| SELL | retest2 | 2026-02-12 09:15:00 | 26580.00 | 2026-02-17 13:15:00 | 26365.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2026-02-26 15:15:00 | 26850.00 | 2026-02-27 09:15:00 | 26380.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-03-04 09:15:00 | 25200.00 | 2026-03-09 10:15:00 | 23940.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 25200.00 | 2026-03-10 10:15:00 | 24170.00 | STOP_HIT | 0.50 | 4.09% |
| BUY | retest2 | 2026-03-27 13:15:00 | 23850.00 | 2026-03-27 14:15:00 | 23620.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-09 12:15:00 | 24060.00 | 2026-04-09 15:15:00 | 23755.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-10 09:15:00 | 24185.00 | 2026-04-23 12:15:00 | 25325.00 | STOP_HIT | 1.00 | 4.71% |
| BUY | retest2 | 2026-04-13 10:15:00 | 24000.00 | 2026-04-23 12:15:00 | 25325.00 | STOP_HIT | 1.00 | 5.52% |
| SELL | retest2 | 2026-04-28 11:00:00 | 24825.00 | 2026-05-04 15:15:00 | 24730.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-04-28 15:00:00 | 24840.00 | 2026-05-04 15:15:00 | 24730.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2026-04-29 13:15:00 | 24785.00 | 2026-05-04 15:15:00 | 24730.00 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2026-05-04 11:30:00 | 24830.00 | 2026-05-04 15:15:00 | 24730.00 | STOP_HIT | 1.00 | 0.40% |
