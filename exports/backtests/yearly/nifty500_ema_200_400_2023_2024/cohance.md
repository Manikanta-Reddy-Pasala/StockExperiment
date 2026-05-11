# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 487.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 63 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 56
- **Target hits / Stop hits / Partials:** 3 / 61 / 11
- **Avg / median % per leg:** -0.79% / -1.47%
- **Sum % (uncompounded):** -59.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 2 | 11 | 0 | -0.22% | -2.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.75% | -0.8% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 2 | 10 | 0 | -0.18% | -2.2% |
| SELL (all) | 62 | 17 | 27.4% | 1 | 50 | 11 | -0.91% | -56.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 17 | 27.4% | 1 | 50 | 11 | -0.91% | -56.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.75% | -0.8% |
| retest2 (combined) | 74 | 19 | 25.7% | 3 | 60 | 11 | -0.79% | -58.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 12:15:00 | 481.65 | 474.34 | 474.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 483.35 | 475.18 | 474.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 15:15:00 | 480.00 | 481.01 | 478.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:15:00 | 484.35 | 481.01 | 478.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 480.70 | 483.97 | 481.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-20 12:15:00 | 480.70 | 483.97 | 481.11 | SL hit (close<ema400) qty=1.00 sl=481.11 alert=retest1 |

### Cycle 2 — SELL (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 12:15:00 | 625.60 | 660.27 | 660.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 14:15:00 | 615.75 | 659.48 | 660.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 14:15:00 | 651.75 | 649.16 | 654.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 651.75 | 649.16 | 654.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 646.00 | 649.13 | 654.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:15:00 | 642.25 | 649.13 | 654.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 10:15:00 | 644.45 | 649.10 | 654.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 11:45:00 | 644.35 | 649.04 | 654.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 12:45:00 | 645.15 | 649.01 | 654.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 650.05 | 648.33 | 653.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 650.05 | 648.33 | 653.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 654.65 | 648.39 | 653.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:00:00 | 654.65 | 648.39 | 653.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 651.25 | 648.42 | 653.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 14:00:00 | 646.05 | 648.44 | 653.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 688.65 | 648.84 | 653.56 | SL hit (close>static) qty=1.00 sl=655.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 660.75 | 651.51 | 651.46 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 09:15:00 | 638.90 | 651.33 | 651.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 13:15:00 | 636.90 | 650.38 | 650.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 649.50 | 643.97 | 647.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 649.50 | 643.97 | 647.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 649.50 | 643.97 | 647.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 649.50 | 643.97 | 647.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 650.70 | 644.03 | 647.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 643.20 | 644.31 | 647.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:30:00 | 645.35 | 644.28 | 647.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 611.04 | 643.65 | 646.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 613.08 | 643.65 | 646.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 639.40 | 637.54 | 643.05 | SL hit (close>ema200) qty=0.50 sl=637.54 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 691.70 | 647.81 | 647.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 14:15:00 | 715.95 | 650.49 | 648.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 10:15:00 | 1247.45 | 1247.50 | 1176.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 10:45:00 | 1248.40 | 1247.50 | 1176.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1267.85 | 1276.65 | 1236.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:15:00 | 1283.95 | 1275.13 | 1237.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 1211.50 | 1271.14 | 1237.67 | SL hit (close<static) qty=1.00 sl=1220.05 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 1114.25 | 1213.12 | 1213.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 1095.20 | 1200.43 | 1206.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 1126.60 | 1077.62 | 1122.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 14:15:00 | 1126.60 | 1077.62 | 1122.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1126.60 | 1077.62 | 1122.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 1126.60 | 1077.62 | 1122.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 1099.90 | 1077.84 | 1122.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 1142.90 | 1077.84 | 1122.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1119.00 | 1078.25 | 1122.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1104.15 | 1078.25 | 1122.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 1113.60 | 1095.50 | 1124.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 12:00:00 | 1113.25 | 1095.68 | 1124.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 15:15:00 | 1110.70 | 1096.32 | 1124.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1048.94 | 1095.35 | 1122.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1057.92 | 1095.35 | 1122.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1057.59 | 1095.35 | 1122.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1055.16 | 1095.35 | 1122.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1145.00 | 1093.33 | 1120.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 1145.00 | 1093.33 | 1120.82 | SL hit (close>ema200) qty=0.50 sl=1093.33 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1177.50 | 1136.28 | 1136.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1182.85 | 1136.74 | 1136.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 1138.60 | 1143.28 | 1140.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 1138.60 | 1143.28 | 1140.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1138.60 | 1143.28 | 1140.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:45:00 | 1136.30 | 1143.28 | 1140.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1148.60 | 1143.33 | 1140.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:00:00 | 1152.00 | 1143.42 | 1140.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 15:00:00 | 1153.65 | 1143.52 | 1140.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 1151.05 | 1143.76 | 1140.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 1134.70 | 1143.66 | 1140.37 | SL hit (close<static) qty=1.00 sl=1136.90 alert=retest2 |

### Cycle 8 — SELL (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 11:15:00 | 1100.05 | 1139.96 | 1139.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 13:15:00 | 1093.10 | 1139.08 | 1139.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 10:15:00 | 1121.20 | 1120.65 | 1129.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-11 11:00:00 | 1121.20 | 1120.65 | 1129.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 1132.60 | 1120.76 | 1129.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:00:00 | 1132.60 | 1120.76 | 1129.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1119.65 | 1120.75 | 1129.43 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 15:15:00 | 1201.00 | 1137.28 | 1137.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1243.20 | 1138.33 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 14:15:00 | 1144.20 | 1152.79 | 1145.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 14:15:00 | 1144.20 | 1152.79 | 1145.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1144.20 | 1152.79 | 1145.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:30:00 | 1143.20 | 1152.79 | 1145.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 1145.00 | 1152.71 | 1145.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 1145.70 | 1152.71 | 1145.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 1150.30 | 1152.77 | 1145.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 1148.40 | 1152.77 | 1145.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 1147.90 | 1152.72 | 1145.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 1149.00 | 1152.72 | 1145.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1142.00 | 1152.58 | 1145.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 1142.00 | 1152.58 | 1145.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1154.00 | 1152.60 | 1145.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 1135.90 | 1152.43 | 1145.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1140.20 | 1152.30 | 1145.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 1137.20 | 1152.30 | 1145.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1147.70 | 1151.45 | 1145.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 1141.80 | 1151.45 | 1145.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 1138.70 | 1151.32 | 1145.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:30:00 | 1138.10 | 1151.32 | 1145.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 1145.90 | 1151.27 | 1145.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 13:30:00 | 1150.00 | 1151.22 | 1145.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 14:15:00 | 1137.90 | 1151.09 | 1145.32 | SL hit (close<static) qty=1.00 sl=1138.60 alert=retest2 |

### Cycle 10 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1059.20 | 1140.22 | 1140.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 1044.00 | 1115.21 | 1126.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 1109.30 | 1100.86 | 1115.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 1109.30 | 1100.86 | 1115.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1109.30 | 1100.86 | 1115.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 1112.60 | 1100.86 | 1115.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1040.70 | 1011.55 | 1045.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 1040.70 | 1011.55 | 1045.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1032.10 | 1011.96 | 1045.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 1023.85 | 1011.96 | 1045.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 1024.45 | 1012.09 | 1045.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 1023.55 | 1009.47 | 1039.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1022.50 | 1010.08 | 1038.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1055.65 | 1010.66 | 1038.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1055.65 | 1010.66 | 1038.13 | SL hit (close>static) qty=1.00 sl=1049.90 alert=retest2 |

### Cycle 11 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 474.40 | 374.01 | 373.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 483.75 | 383.33 | 378.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 09:15:00 | 473.00 | 2023-05-16 09:15:00 | 476.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-05-17 09:15:00 | 473.10 | 2023-05-18 10:15:00 | 477.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-05-17 11:00:00 | 473.50 | 2023-05-18 10:15:00 | 477.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-05-18 09:15:00 | 474.00 | 2023-05-18 10:15:00 | 477.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-05-19 09:15:00 | 472.55 | 2023-05-22 10:15:00 | 475.60 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-05-22 09:15:00 | 473.00 | 2023-05-22 10:15:00 | 475.60 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-05-22 14:30:00 | 474.40 | 2023-05-22 15:15:00 | 476.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2023-05-23 14:15:00 | 474.10 | 2023-05-26 15:15:00 | 477.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-05-23 14:45:00 | 474.05 | 2023-05-26 15:15:00 | 477.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-05-23 15:15:00 | 473.90 | 2023-05-26 15:15:00 | 477.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-05-25 13:45:00 | 473.50 | 2023-06-02 12:15:00 | 481.65 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-05-26 09:15:00 | 474.05 | 2023-06-02 12:15:00 | 481.65 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-05-26 09:45:00 | 473.80 | 2023-06-02 12:15:00 | 481.65 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-05-26 11:00:00 | 474.35 | 2023-06-02 12:15:00 | 481.65 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-06-01 13:45:00 | 473.10 | 2023-06-02 12:15:00 | 481.65 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest1 | 2023-07-04 09:15:00 | 484.35 | 2023-07-20 12:15:00 | 480.70 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-07-21 14:45:00 | 484.65 | 2023-08-17 09:15:00 | 533.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-25 09:30:00 | 487.00 | 2023-09-13 14:15:00 | 535.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-26 09:15:00 | 642.25 | 2024-03-28 14:15:00 | 688.65 | STOP_HIT | 1.00 | -7.22% |
| SELL | retest2 | 2024-03-26 10:15:00 | 644.45 | 2024-03-28 14:15:00 | 688.65 | STOP_HIT | 1.00 | -6.86% |
| SELL | retest2 | 2024-03-26 11:45:00 | 644.35 | 2024-03-28 14:15:00 | 688.65 | STOP_HIT | 1.00 | -6.88% |
| SELL | retest2 | 2024-03-26 12:45:00 | 645.15 | 2024-03-28 14:15:00 | 688.65 | STOP_HIT | 1.00 | -6.74% |
| SELL | retest2 | 2024-03-28 14:00:00 | 646.05 | 2024-03-28 14:15:00 | 688.65 | STOP_HIT | 1.00 | -6.59% |
| SELL | retest2 | 2024-04-04 11:30:00 | 648.70 | 2024-04-09 09:15:00 | 616.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-04 12:45:00 | 648.95 | 2024-04-09 09:15:00 | 616.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-04 13:30:00 | 647.90 | 2024-04-09 09:15:00 | 615.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 11:30:00 | 638.15 | 2024-04-15 09:15:00 | 606.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-04 11:30:00 | 648.70 | 2024-04-18 12:15:00 | 647.50 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2024-04-04 12:45:00 | 648.95 | 2024-04-18 12:15:00 | 647.50 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2024-04-04 13:30:00 | 647.90 | 2024-04-18 12:15:00 | 647.50 | STOP_HIT | 0.50 | 0.06% |
| SELL | retest2 | 2024-04-08 11:30:00 | 638.15 | 2024-04-18 12:15:00 | 647.50 | STOP_HIT | 0.50 | -1.47% |
| SELL | retest2 | 2024-04-08 12:15:00 | 634.75 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-04-18 09:30:00 | 628.15 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2024-04-19 09:45:00 | 638.05 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-04-22 11:00:00 | 640.65 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-04-22 11:45:00 | 640.60 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-04-23 14:00:00 | 640.00 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-04-25 09:15:00 | 640.15 | 2024-04-26 09:15:00 | 659.30 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-04-29 10:15:00 | 640.80 | 2024-04-29 14:15:00 | 653.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-05-13 13:15:00 | 640.00 | 2024-05-13 14:15:00 | 663.65 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-05-13 13:45:00 | 641.00 | 2024-05-13 14:15:00 | 663.65 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-05-29 15:15:00 | 643.20 | 2024-05-31 09:15:00 | 611.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 09:30:00 | 645.35 | 2024-05-31 09:15:00 | 613.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 15:15:00 | 643.20 | 2024-06-06 10:15:00 | 639.40 | STOP_HIT | 0.50 | 0.59% |
| SELL | retest2 | 2024-05-30 09:30:00 | 645.35 | 2024-06-06 10:15:00 | 639.40 | STOP_HIT | 0.50 | 0.92% |
| BUY | retest2 | 2024-12-18 13:15:00 | 1283.95 | 2024-12-19 15:15:00 | 1211.50 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1104.15 | 2025-02-12 09:15:00 | 1048.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1113.60 | 2025-02-12 09:15:00 | 1057.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 12:00:00 | 1113.25 | 2025-02-12 09:15:00 | 1057.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 15:15:00 | 1110.70 | 2025-02-12 09:15:00 | 1055.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1104.15 | 2025-02-13 09:15:00 | 1145.00 | STOP_HIT | 0.50 | -3.70% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1113.60 | 2025-02-13 09:15:00 | 1145.00 | STOP_HIT | 0.50 | -2.82% |
| SELL | retest2 | 2025-02-10 12:00:00 | 1113.25 | 2025-02-13 09:15:00 | 1145.00 | STOP_HIT | 0.50 | -2.85% |
| SELL | retest2 | 2025-02-10 15:15:00 | 1110.70 | 2025-02-13 09:15:00 | 1145.00 | STOP_HIT | 0.50 | -3.09% |
| SELL | retest2 | 2025-02-13 14:15:00 | 1120.30 | 2025-02-17 10:15:00 | 1145.50 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-02-17 12:00:00 | 1116.25 | 2025-02-18 10:15:00 | 1126.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-02-17 14:30:00 | 1119.95 | 2025-02-19 09:15:00 | 1128.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-17 15:00:00 | 1117.00 | 2025-02-19 10:15:00 | 1145.20 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-02-18 09:30:00 | 1100.00 | 2025-02-19 10:15:00 | 1145.20 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-02-18 11:30:00 | 1100.30 | 2025-02-19 10:15:00 | 1145.20 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-03-12 14:00:00 | 1152.00 | 2025-03-13 11:15:00 | 1134.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-03-12 15:00:00 | 1153.65 | 2025-03-13 11:15:00 | 1134.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-03-13 10:15:00 | 1151.05 | 2025-03-13 11:15:00 | 1134.70 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-03-13 15:15:00 | 1151.85 | 2025-03-17 14:15:00 | 1132.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-03-19 15:15:00 | 1167.75 | 2025-03-26 09:15:00 | 1101.30 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2025-04-30 13:30:00 | 1150.00 | 2025-04-30 14:15:00 | 1137.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-02 09:45:00 | 1149.90 | 2025-05-02 11:15:00 | 1138.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-05-05 09:45:00 | 1150.70 | 2025-05-05 12:15:00 | 1136.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-05 15:00:00 | 1150.70 | 2025-05-05 15:15:00 | 1136.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-07 09:15:00 | 1023.85 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-07-07 10:00:00 | 1024.45 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-07-11 10:30:00 | 1023.55 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1022.50 | 2025-07-15 09:15:00 | 1055.65 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-07-24 13:45:00 | 1039.05 | 2025-07-29 10:15:00 | 987.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:45:00 | 1039.05 | 2025-08-06 10:15:00 | 935.14 | TARGET_HIT | 0.50 | 10.00% |
