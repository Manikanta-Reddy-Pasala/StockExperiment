# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1487.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 17
- **Target hits / Stop hits / Partials:** 3 / 17 / 3
- **Avg / median % per leg:** 0.69% / -1.39%
- **Sum % (uncompounded):** 15.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.46% | -13.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.46% | -13.2% |
| SELL (all) | 14 | 6 | 42.9% | 3 | 8 | 3 | 2.07% | 29.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 3 | 8 | 3 | 2.07% | 29.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 6 | 26.1% | 3 | 17 | 3 | 0.69% | 15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1137.80 | 1177.62 | 1177.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 1134.50 | 1168.35 | 1172.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 1132.50 | 1141.33 | 1155.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 1128.40 | 1141.12 | 1155.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 1162.40 | 1140.58 | 1154.01 | SL hit (close>static) qty=1.00 sl=1159.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1103.20 | 1097.75 | 1097.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1107.90 | 1097.90 | 1097.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1097.50 | 1098.81 | 1098.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1100.00 | 1098.82 | 1098.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1098.60 | 1098.82 | 1098.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1103.50 | 1098.86 | 1098.32 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1089.00 | 1097.74 | 1097.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1084.60 | 1097.61 | 1097.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1097.80 | 1097.40 | 1097.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1103.50 | 1097.46 | 1097.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1107.30 | 1097.46 | 1097.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1109.90 | 1097.58 | 1097.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1109.90 | 1097.58 | 1097.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1113.70 | 1097.93 | 1097.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1123.70 | 1099.09 | 1098.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1187.00 | 1189.47 | 1160.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1187.00 | 1189.47 | 1160.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1160.50 | 1187.69 | 1160.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1177.70 | 1187.69 | 1160.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 1167.70 | 1200.15 | 1183.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1160.40 | 1199.76 | 1183.86 | SL hit (close<static) qty=1.00 sl=1160.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1173.45 | 1173.54 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 1191.20 | 1173.66 | 1173.61 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1157.40 | 1173.41 | 1173.48 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1228.60 | 1173.71 | 1173.63 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1131.20 | 1174.73 | 1174.75 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1222.50 | 1174.56 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1246.90 | 1190.02 | 1183.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-23 14:45:00 | 1197.30 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1201.40 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-26 10:45:00 | 1198.50 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-27 09:30:00 | 1203.40 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-27 15:00:00 | 1191.90 | 2025-05-28 09:15:00 | 1167.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-30 12:30:00 | 1132.50 | 2025-07-02 12:15:00 | 1162.40 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-01 09:15:00 | 1128.40 | 2025-07-02 12:15:00 | 1162.40 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-07-10 11:00:00 | 1131.60 | 2025-07-17 10:15:00 | 1157.90 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-10 12:45:00 | 1131.30 | 2025-07-17 11:15:00 | 1161.20 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1146.50 | 2025-07-17 11:15:00 | 1161.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1143.40 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-07-18 13:00:00 | 1143.70 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-21 12:45:00 | 1145.30 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1147.00 | 2025-08-01 09:15:00 | 1089.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1150.60 | 2025-08-01 09:15:00 | 1093.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1148.10 | 2025-08-01 09:15:00 | 1090.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1147.00 | 2025-08-11 09:15:00 | 1032.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1150.60 | 2025-08-11 09:15:00 | 1035.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1148.10 | 2025-08-11 09:15:00 | 1033.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1177.70 | 2026-01-13 11:15:00 | 1160.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-13 10:45:00 | 1167.70 | 2026-01-13 11:15:00 | 1160.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-01-13 13:00:00 | 1166.10 | 2026-01-20 09:15:00 | 1158.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-13 14:45:00 | 1169.10 | 2026-01-20 09:15:00 | 1158.90 | STOP_HIT | 1.00 | -0.87% |
