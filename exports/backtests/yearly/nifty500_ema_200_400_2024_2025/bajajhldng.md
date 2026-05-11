# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 49 |
| PARTIAL | 8 |
| TARGET_HIT | 19 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 29
- **Target hits / Stop hits / Partials:** 19 / 34 / 8
- **Avg / median % per leg:** 2.75% / 0.29%
- **Sum % (uncompounded):** 167.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 22 | 53.7% | 18 | 19 | 4 | 3.66% | 150.0% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 33 | 14 | 42.4% | 14 | 19 | 0 | 2.73% | 90.0% |
| SELL (all) | 20 | 10 | 50.0% | 1 | 15 | 4 | 0.88% | 17.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 10 | 50.0% | 1 | 15 | 4 | 0.88% | 17.6% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 53 | 24 | 45.3% | 15 | 34 | 4 | 2.03% | 107.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 8460.00 | 8267.64 | 8266.96 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-21 13:15:00)

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

### Cycle 3 — BUY (started 2024-06-24 11:15:00)

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

### Cycle 4 — SELL (started 2025-09-02 11:15:00)

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
