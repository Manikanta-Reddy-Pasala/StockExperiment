# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1176.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 15
- **Target hits / Stop hits / Partials:** 0 / 15 / 0
- **Avg / median % per leg:** -1.50% / -1.22%
- **Sum % (uncompounded):** -22.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.26% | -16.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.26% | -16.3% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.07% | -6.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.07% | -6.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.50% | -22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 1064.00 | 1093.61 | 1093.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1056.60 | 1075.64 | 1080.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1090.10 | 1074.79 | 1079.45 | SL hit (close>static) qty=1.00 sl=1090.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.90 | 1082.45 | 1082.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.89 | 1083.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1114.90 | 1114.92 | 1104.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:45:00 | 1113.70 | 1114.92 | 1104.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.30 | 1162.92 | 1147.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1149.30 | 1162.92 | 1147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1144.20 | 1162.73 | 1147.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 1143.80 | 1162.73 | 1147.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1139.10 | 1162.49 | 1147.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 1135.80 | 1162.49 | 1147.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1142.80 | 1161.18 | 1146.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1142.80 | 1161.18 | 1146.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1141.80 | 1160.98 | 1146.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1141.80 | 1160.98 | 1146.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1148.20 | 1160.52 | 1146.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1151.60 | 1160.14 | 1146.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1137.60 | 1159.42 | 1147.29 | SL hit (close<static) qty=1.00 sl=1143.10 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 14:15:00 | 1124.50 | 1161.68 | 1161.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:45:00 | 1081.90 | 1076.96 | 1104.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 1092.00 | 1079.78 | 1102.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1124.50 | 1081.55 | 1102.40 | SL hit (close>static) qty=1.00 sl=1108.30 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1116.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.63 | 1117.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 11:00:00 | 1101.50 | 2025-07-01 10:15:00 | 1086.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-30 10:30:00 | 1101.70 | 2025-07-01 10:15:00 | 1086.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-03 10:00:00 | 1101.80 | 2025-07-03 11:15:00 | 1093.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-07 11:30:00 | 1103.20 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.50 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-09 12:00:00 | 1105.80 | 2025-07-10 14:15:00 | 1089.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-17 13:45:00 | 1107.80 | 2025-07-18 12:15:00 | 1095.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-18 09:30:00 | 1104.80 | 2025-07-18 12:15:00 | 1095.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1056.60 | 2025-09-02 09:15:00 | 1090.10 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-12-05 10:45:00 | 1151.60 | 2025-12-09 09:15:00 | 1137.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-10 10:00:00 | 1155.00 | 2025-12-10 12:15:00 | 1141.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1151.60 | 2025-12-11 14:15:00 | 1141.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-12 15:00:00 | 1150.70 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-15 10:30:00 | 1160.60 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-04-15 15:15:00 | 1092.00 | 2026-04-17 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -2.98% |
