# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1248.50
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
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 14 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 0
- **Target hits / Stop hits / Partials:** 14 / 0 / 0
- **Avg / median % per leg:** 10.00% / 10.00%
- **Sum % (uncompounded):** 140.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 14 | 100.0% | 14 | 0 | 0 | 10.00% | 140.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 14 | 100.0% | 14 | 0 | 0 | 10.00% | 140.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 14 | 100.0% | 14 | 0 | 0 | 10.00% | 140.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 927.75 | 884.09 | 884.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 948.70 | 885.54 | 884.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 937.20 | 942.23 | 923.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 937.20 | 942.23 | 923.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 933.00 | 942.08 | 923.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 934.95 | 927.54 | 919.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 936.30 | 927.55 | 920.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:00:00 | 934.70 | 939.65 | 929.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 937.35 | 939.48 | 929.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 928.15 | 939.31 | 930.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 928.15 | 939.31 | 930.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 928.95 | 939.21 | 930.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 928.05 | 939.21 | 930.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 932.20 | 939.14 | 930.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 935.90 | 939.14 | 930.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 939.50 | 939.11 | 930.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:30:00 | 935.15 | 939.26 | 930.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 938.10 | 939.19 | 930.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 933.00 | 939.08 | 930.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 931.25 | 939.08 | 930.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 944.70 | 958.23 | 944.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 944.70 | 958.23 | 944.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 946.95 | 958.12 | 944.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 967.00 | 958.12 | 944.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:45:00 | 948.30 | 980.31 | 965.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 949.20 | 980.31 | 965.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-03 14:15:00 | 1028.45 | 980.31 | 967.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1012.70 | 1031.40 | 1031.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1007.80 | 1030.95 | 1031.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 1022.40 | 1016.55 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1022.10 | 1016.61 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 1023.00 | 1016.61 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1020.70 | 1016.65 | 1022.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1059.90 | 1016.65 | 1022.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1052.90 | 1017.01 | 1022.97 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 1077.80 | 1028.55 | 1028.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 1082.30 | 1029.09 | 1028.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1034.40 | 1036.05 | 1032.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1031.60 | 1036.00 | 1032.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1031.60 | 1036.00 | 1032.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1032.00 | 1035.97 | 1032.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 1035.80 | 1031.33 | 1030.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1039.30 | 1035.13 | 1032.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 14:30:00 | 1037.70 | 1035.25 | 1032.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-29 09:15:00 | 1139.38 | 1048.19 | 1039.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 11:15:00 | 934.95 | 2025-09-03 14:15:00 | 1028.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 12:45:00 | 936.30 | 2025-09-03 14:15:00 | 1029.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 11:00:00 | 934.70 | 2025-09-03 14:15:00 | 1028.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 13:45:00 | 937.35 | 2025-09-03 14:15:00 | 1029.49 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2025-07-14 15:15:00 | 935.90 | 2025-09-03 14:15:00 | 1028.66 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2025-07-15 09:45:00 | 939.50 | 2025-09-04 09:15:00 | 1031.09 | TARGET_HIT | 1.00 | 9.75% |
| BUY | retest2 | 2025-07-16 14:30:00 | 935.15 | 2025-09-04 09:15:00 | 1033.45 | TARGET_HIT | 1.00 | 10.51% |
| BUY | retest2 | 2025-07-17 09:15:00 | 938.10 | 2025-09-04 09:15:00 | 1031.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 967.00 | 2025-09-08 09:15:00 | 1043.13 | TARGET_HIT | 1.00 | 7.87% |
| BUY | retest2 | 2025-08-29 14:45:00 | 948.30 | 2025-09-08 09:15:00 | 1044.12 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2025-08-29 15:15:00 | 949.20 | 2025-09-23 13:15:00 | 1063.70 | TARGET_HIT | 1.00 | 12.06% |
| BUY | retest2 | 2026-01-14 10:30:00 | 1035.80 | 2026-01-29 09:15:00 | 1139.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 11:00:00 | 1039.30 | 2026-01-29 09:15:00 | 1143.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 14:30:00 | 1037.70 | 2026-01-29 09:15:00 | 1141.47 | TARGET_HIT | 1.00 | 10.00% |
