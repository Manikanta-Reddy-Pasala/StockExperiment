# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 89 |
| PARTIAL | 8 |
| TARGET_HIT | 23 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 65
- **Target hits / Stop hits / Partials:** 23 / 70 / 8
- **Avg / median % per leg:** 1.35% / -1.09%
- **Sum % (uncompounded):** 136.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 26 | 40.0% | 22 | 39 | 4 | 2.35% | 153.0% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 57 | 18 | 31.6% | 18 | 39 | 0 | 1.63% | 93.0% |
| SELL (all) | 36 | 10 | 27.8% | 1 | 31 | 4 | -0.47% | -16.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 10 | 27.8% | 1 | 31 | 4 | -0.47% | -16.9% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 93 | 28 | 30.1% | 19 | 70 | 4 | 0.82% | 76.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 6738.85 | 7169.71 | 7171.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 14:15:00 | 6662.50 | 7164.66 | 7168.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 7060.00 | 7003.49 | 7072.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 7060.00 | 7003.49 | 7072.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 7060.00 | 7003.49 | 7072.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:45:00 | 7100.95 | 7003.49 | 7072.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 7042.05 | 7003.87 | 7072.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 7008.15 | 7003.58 | 7071.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:45:00 | 7006.75 | 7003.24 | 7070.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 14:30:00 | 7013.00 | 7003.74 | 7070.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 7096.85 | 7005.79 | 7070.65 | SL hit (close>static) qty=1.00 sl=7088.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 14:15:00 | 7347.00 | 7059.79 | 7058.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 7409.60 | 7065.83 | 7061.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 7694.95 | 7698.31 | 7488.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 15:00:00 | 7694.95 | 7698.31 | 7488.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 8367.45 | 8631.74 | 8397.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 15:00:00 | 8367.45 | 8631.74 | 8397.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 8398.00 | 8629.41 | 8397.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 8169.65 | 8629.41 | 8397.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 8215.00 | 8625.29 | 8396.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:30:00 | 8125.75 | 8625.29 | 8396.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 8361.00 | 8474.71 | 8357.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 11:15:00 | 8425.00 | 8474.71 | 8357.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 14:30:00 | 8396.80 | 8471.18 | 8358.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-26 09:15:00 | 8264.20 | 8468.00 | 8357.80 | SL hit (close<static) qty=1.00 sl=8304.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 13:15:00 | 7904.90 | 8302.00 | 8303.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 7822.00 | 8265.29 | 8284.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 8289.65 | 8222.17 | 8260.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 11:15:00 | 8289.65 | 8222.17 | 8260.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 8289.65 | 8222.17 | 8260.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:45:00 | 8298.30 | 8222.17 | 8260.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 8177.10 | 8221.72 | 8259.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 14:15:00 | 8129.75 | 8221.04 | 8259.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 8129.00 | 8217.94 | 8254.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 10:15:00 | 8131.35 | 8217.22 | 8254.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 12:15:00 | 8138.90 | 8215.90 | 8253.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 8250.65 | 8213.03 | 8250.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 12:45:00 | 8174.65 | 8212.91 | 8250.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 14:15:00 | 8180.80 | 8212.68 | 8249.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 10:45:00 | 8155.20 | 8211.54 | 8248.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:30:00 | 8170.00 | 8201.18 | 8239.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 8266.30 | 8200.90 | 8238.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:45:00 | 8260.00 | 8200.90 | 8238.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 8250.00 | 8201.39 | 8238.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:30:00 | 8239.80 | 8201.88 | 8238.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 14:15:00 | 8276.55 | 8203.58 | 8238.47 | SL hit (close>static) qty=1.00 sl=8267.65 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 8460.00 | 8267.64 | 8266.96 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 8130.70 | 8266.84 | 8266.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 8103.60 | 8260.06 | 8263.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 8126.45 | 8110.96 | 8174.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 8126.45 | 8110.96 | 8174.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 8321.60 | 8114.32 | 8174.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 8277.55 | 8114.32 | 8174.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 8328.90 | 8116.46 | 8175.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 13:30:00 | 8275.90 | 8121.68 | 8176.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 14:15:00 | 8393.20 | 8124.38 | 8177.92 | SL hit (close>static) qty=1.00 sl=8333.45 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 8750.00 | 8221.25 | 8219.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 8880.15 | 8374.65 | 8304.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 9315.95 | 9375.71 | 9035.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:30:00 | 9387.75 | 9356.71 | 9050.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:30:00 | 9414.60 | 9357.05 | 9051.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 13:15:00 | 9400.00 | 9357.28 | 9053.58 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 14:00:00 | 9401.55 | 9357.72 | 9055.31 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 12:15:00 | 9857.14 | 9446.06 | 9191.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 12:15:00 | 9870.00 | 9446.06 | 9191.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 14:15:00 | 9885.33 | 9454.43 | 9197.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 14:15:00 | 9871.63 | 9454.43 | 9197.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-28 09:15:00 | 10326.53 | 9547.61 | 9274.91 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 7 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 12620.00 | 13595.83 | 13597.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 12599.00 | 13558.31 | 13578.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 13358.00 | 13309.78 | 13424.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 13358.00 | 13309.78 | 13424.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 13545.00 | 13312.12 | 13424.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 13545.00 | 13312.12 | 13424.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 13550.00 | 13314.49 | 13425.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 13702.00 | 13314.49 | 13425.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 13100.00 | 12600.57 | 12898.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 13100.00 | 12600.57 | 12898.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 13114.00 | 12605.68 | 12899.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 13013.00 | 12631.23 | 12905.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 13029.00 | 12635.45 | 12905.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:45:00 | 13055.00 | 12639.56 | 12906.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 13044.00 | 12643.59 | 12907.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 13155.00 | 12656.25 | 12909.65 | SL hit (close>static) qty=1.00 sl=13150.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-19 10:30:00 | 6341.70 | 2023-05-26 10:15:00 | 6975.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 11:00:00 | 6339.80 | 2023-05-26 10:15:00 | 6973.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 11:30:00 | 6340.20 | 2023-05-26 10:15:00 | 6974.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 14:00:00 | 6348.20 | 2023-05-26 10:15:00 | 6983.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 09:15:00 | 7172.50 | 2023-08-18 09:15:00 | 7072.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-08-24 11:15:00 | 7170.40 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-08-24 12:00:00 | 7168.00 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-08-24 13:15:00 | 7174.95 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-08-25 11:30:00 | 7230.00 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-08-25 13:30:00 | 7225.05 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2023-08-25 14:15:00 | 7229.90 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-08-25 14:45:00 | 7226.95 | 2023-09-12 10:15:00 | 7092.85 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-09-18 10:30:00 | 7245.65 | 2023-09-28 14:15:00 | 7051.65 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2023-09-18 12:45:00 | 7240.40 | 2023-09-28 14:15:00 | 7051.65 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2023-09-20 09:30:00 | 7265.00 | 2023-09-28 14:15:00 | 7051.65 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2023-09-20 15:15:00 | 7250.00 | 2023-09-28 14:15:00 | 7051.65 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2023-10-20 11:45:00 | 7008.15 | 2023-10-23 10:15:00 | 7096.85 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-10-20 13:45:00 | 7006.75 | 2023-10-23 10:15:00 | 7096.85 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-10-20 14:30:00 | 7013.00 | 2023-10-23 10:15:00 | 7096.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-10-25 09:15:00 | 6973.95 | 2023-11-08 09:15:00 | 7120.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-10-25 14:30:00 | 6890.25 | 2023-11-08 14:15:00 | 7129.85 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2023-11-03 12:00:00 | 6906.80 | 2023-11-08 14:15:00 | 7129.85 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2023-11-03 13:30:00 | 6918.55 | 2023-11-08 14:15:00 | 7129.85 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-03-22 11:15:00 | 8425.00 | 2024-03-26 09:15:00 | 8264.20 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-03-22 14:30:00 | 8396.80 | 2024-03-26 09:15:00 | 8264.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-03-26 13:30:00 | 8405.55 | 2024-03-27 14:15:00 | 8051.30 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-04-01 09:15:00 | 8385.05 | 2024-04-03 15:15:00 | 8330.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-04-01 14:15:00 | 8453.95 | 2024-04-03 15:15:00 | 8330.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-02 09:15:00 | 8430.45 | 2024-04-03 15:15:00 | 8330.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-04-02 13:45:00 | 8425.00 | 2024-04-03 15:15:00 | 8330.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-04-02 15:15:00 | 8437.45 | 2024-04-04 09:15:00 | 8286.15 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-04-23 14:15:00 | 8129.75 | 2024-05-07 14:15:00 | 8276.55 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-04-26 09:15:00 | 8129.00 | 2024-05-08 09:15:00 | 8328.95 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-04-26 10:15:00 | 8131.35 | 2024-05-08 09:15:00 | 8328.95 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-04-26 12:15:00 | 8138.90 | 2024-05-08 09:15:00 | 8328.95 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-04-29 12:45:00 | 8174.65 | 2024-05-08 09:15:00 | 8328.95 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-04-29 14:15:00 | 8180.80 | 2024-05-08 14:15:00 | 8350.55 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-04-30 10:45:00 | 8155.20 | 2024-05-08 14:15:00 | 8350.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-05-06 09:30:00 | 8170.00 | 2024-05-08 14:15:00 | 8350.55 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-05-07 11:30:00 | 8239.80 | 2024-05-08 14:15:00 | 8350.55 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-06-07 13:30:00 | 8275.90 | 2024-06-07 14:15:00 | 8393.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-06-13 14:45:00 | 8270.20 | 2024-06-18 09:15:00 | 8348.65 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-06-18 15:15:00 | 8273.60 | 2024-06-21 13:15:00 | 8250.00 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-06-19 09:30:00 | 8271.85 | 2024-06-21 13:15:00 | 8250.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-06-19 12:15:00 | 8199.05 | 2024-06-21 13:15:00 | 8250.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-06-19 15:00:00 | 8199.95 | 2024-06-24 09:15:00 | 8606.80 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2024-06-20 09:30:00 | 8175.00 | 2024-06-24 09:15:00 | 8606.80 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest1 | 2024-08-07 10:30:00 | 9387.75 | 2024-08-22 12:15:00 | 9857.14 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-07 11:30:00 | 9414.60 | 2024-08-22 12:15:00 | 9870.00 | PARTIAL | 0.50 | 4.84% |
| BUY | retest1 | 2024-08-07 13:15:00 | 9400.00 | 2024-08-22 14:15:00 | 9885.33 | PARTIAL | 0.50 | 5.16% |
| BUY | retest1 | 2024-08-07 14:00:00 | 9401.55 | 2024-08-22 14:15:00 | 9871.63 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-07 10:30:00 | 9387.75 | 2024-08-28 09:15:00 | 10326.53 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-08-07 11:30:00 | 9414.60 | 2024-08-28 09:15:00 | 10356.06 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-08-07 13:15:00 | 9400.00 | 2024-08-28 09:15:00 | 10340.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-08-07 14:00:00 | 9401.55 | 2024-08-28 09:15:00 | 10341.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-21 11:30:00 | 10381.45 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-10-21 12:15:00 | 10482.65 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-10-22 09:15:00 | 10365.90 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-10-22 09:45:00 | 10410.90 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-10-23 15:15:00 | 10278.40 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-10-24 09:45:00 | 10221.75 | 2024-10-24 12:15:00 | 10132.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-10-28 10:45:00 | 10219.05 | 2024-10-31 09:15:00 | 10159.65 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-10-29 09:15:00 | 10235.10 | 2024-10-31 09:15:00 | 10159.65 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-29 13:45:00 | 10349.95 | 2024-11-27 11:15:00 | 10230.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-10-29 15:00:00 | 10377.55 | 2024-11-27 11:15:00 | 10230.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-11-01 18:45:00 | 10350.00 | 2024-12-09 15:15:00 | 11240.95 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2024-11-04 09:30:00 | 10448.80 | 2024-12-10 14:15:00 | 11258.61 | TARGET_HIT | 1.00 | 7.75% |
| BUY | retest2 | 2024-11-14 10:45:00 | 11123.30 | 2024-12-11 09:15:00 | 11385.00 | TARGET_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2024-11-14 14:00:00 | 10927.45 | 2024-12-11 09:15:00 | 11493.68 | TARGET_HIT | 1.00 | 5.18% |
| BUY | retest2 | 2024-12-09 10:30:00 | 10873.40 | 2024-12-30 14:15:00 | 11960.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 11:15:00 | 10881.85 | 2024-12-30 14:15:00 | 11970.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-15 11:30:00 | 10865.05 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-15 14:15:00 | 10872.30 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-01-16 09:15:00 | 10942.90 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-01-16 10:15:00 | 10865.00 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-16 13:00:00 | 10961.35 | 2025-02-03 09:15:00 | 11951.56 | TARGET_HIT | 1.00 | 9.03% |
| BUY | retest2 | 2025-01-20 09:15:00 | 10927.90 | 2025-02-03 09:15:00 | 11959.53 | TARGET_HIT | 1.00 | 9.44% |
| BUY | retest2 | 2025-01-20 12:30:00 | 10912.05 | 2025-02-03 09:15:00 | 12037.19 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2025-01-21 09:15:00 | 10999.00 | 2025-02-03 09:15:00 | 11951.50 | TARGET_HIT | 1.00 | 8.66% |
| BUY | retest2 | 2025-01-22 09:15:00 | 11177.50 | 2025-02-03 14:15:00 | 12295.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 11:00:00 | 10870.75 | 2025-04-07 12:15:00 | 10747.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-04-08 12:00:00 | 10998.95 | 2025-04-09 09:15:00 | 10733.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-04-08 12:45:00 | 10869.50 | 2025-04-09 09:15:00 | 10733.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-15 14:30:00 | 11787.00 | 2025-05-08 15:15:00 | 11450.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-04-17 11:00:00 | 11823.00 | 2025-05-08 15:15:00 | 11450.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-05-12 09:45:00 | 11787.00 | 2025-05-16 10:15:00 | 12965.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 10:15:00 | 11849.00 | 2025-05-16 10:15:00 | 13033.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 12:15:00 | 11813.00 | 2025-05-16 10:15:00 | 12994.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 13013.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-24 09:45:00 | 13029.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-24 10:45:00 | 13055.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-24 12:00:00 | 13044.00 | 2025-10-24 14:15:00 | 13155.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-03 09:15:00 | 12237.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 13:15:00 | 12882.00 | 2025-11-06 10:15:00 | 12675.00 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-11-06 12:00:00 | 12882.00 | 2025-11-06 12:15:00 | 13105.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-11 09:15:00 | 12246.45 | PARTIAL | 0.50 | 3.69% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-11 14:15:00 | 12080.20 | PARTIAL | 0.50 | 6.29% |
| SELL | retest2 | 2025-11-07 09:15:00 | 12716.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-10 09:45:00 | 12891.00 | 2025-11-14 12:15:00 | 12539.00 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-17 11:15:00 | 12270.00 | 2025-11-21 10:15:00 | 11656.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 12270.00 | 2025-12-02 09:15:00 | 11043.00 | TARGET_HIT | 0.50 | 10.00% |
