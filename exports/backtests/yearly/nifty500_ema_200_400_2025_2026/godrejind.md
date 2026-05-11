# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1202.00
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
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 13
- **Target hits / Stop hits / Partials:** 1 / 17 / 8
- **Avg / median % per leg:** -1.38% / 0.28%
- **Sum % (uncompounded):** -35.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.48% | -7.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.48% | -7.4% |
| SELL (all) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 13 | 50.0% | 1 | 17 | 8 | -1.38% | -35.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1146.40 | 1182.10 | 1182.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 1126.60 | 1178.84 | 1180.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 1203.40 | 1137.96 | 1154.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1203.40 | 1137.96 | 1154.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1203.40 | 1137.96 | 1154.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1203.40 | 1137.96 | 1154.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1169.10 | 1138.27 | 1154.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 1157.00 | 1139.42 | 1154.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1226.50 | 1142.04 | 1155.43 | SL hit (close>static) qty=1.00 sl=1220.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1286.40 | 1168.60 | 1168.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1299.00 | 1175.58 | 1171.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1198.40 | 1202.77 | 1188.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 1198.40 | 1202.77 | 1188.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1185.10 | 1201.98 | 1188.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1185.10 | 1201.98 | 1188.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1188.20 | 1201.85 | 1188.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1204.70 | 1201.57 | 1188.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 1184.00 | 1201.30 | 1188.54 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1109.00 | 1189.57 | 1189.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1101.00 | 1187.15 | 1188.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1122.20 | 1118.42 | 1143.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:30:00 | 1125.80 | 1118.42 | 1143.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1055.30 | 1020.46 | 1047.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1055.30 | 1020.46 | 1047.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1050.50 | 1020.75 | 1047.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1045.50 | 1020.75 | 1047.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1034.10 | 1021.19 | 1047.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1030.10 | 1021.19 | 1047.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1026.10 | 1021.63 | 1046.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 978.59 | 1014.50 | 1037.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 974.79 | 1014.15 | 1036.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.00 | 998.61 | 1021.99 | SL hit (close>ema200) qty=0.50 sl=998.61 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 1155.05 | 940.22 | 939.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1212.00 | 942.92 | 940.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-14 12:00:00 | 1157.00 | 2025-08-18 10:15:00 | 1226.50 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest2 | 2025-09-04 15:15:00 | 1204.70 | 2025-09-05 11:15:00 | 1184.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-05 13:30:00 | 1197.70 | 2025-09-19 12:15:00 | 1181.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-22 10:15:00 | 1192.90 | 2025-09-22 11:15:00 | 1184.90 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-24 09:45:00 | 1192.40 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-10-01 12:45:00 | 1211.10 | 2025-10-01 14:15:00 | 1186.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-01-21 10:15:00 | 978.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-01-21 11:15:00 | 974.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-13 13:15:00 | 980.88 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-13 14:15:00 | 973.18 | PARTIAL | 0.50 | 5.75% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-13 14:15:00 | 962.25 | PARTIAL | 0.50 | 3.51% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-13 15:15:00 | 960.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-13 15:15:00 | 960.50 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -2.43% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2026-02-12 14:00:00 | 1011.05 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-05 11:15:00 | 928.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-12 09:15:00 | 879.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 968.00 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.81% |
| SELL | retest2 | 2026-04-24 13:00:00 | 977.10 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -23.65% |
| SELL | retest2 | 2026-04-29 09:45:00 | 974.25 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.01% |
