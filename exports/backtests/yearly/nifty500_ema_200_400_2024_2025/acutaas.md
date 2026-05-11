# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2748.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 0
- **Avg / median % per leg:** -0.83% / -1.93%
- **Sum % (uncompounded):** -10.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 2 | 1 | 0 | 4.50% | 13.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 2 | 1 | 0 | 4.50% | 13.5% |
| SELL (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.43% | -24.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.43% | -24.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 2 | 15.4% | 2 | 11 | 0 | -0.83% | -10.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 1098.00 | 1144.34 | 1144.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1075.80 | 1142.51 | 1143.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 1134.30 | 1129.75 | 1136.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 1134.30 | 1129.75 | 1136.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1134.30 | 1129.75 | 1136.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1134.30 | 1129.75 | 1136.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1148.00 | 1129.94 | 1136.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1097.90 | 1129.94 | 1136.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 1115.30 | 1120.84 | 1130.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1160.70 | 1122.77 | 1131.19 | SL hit (close>static) qty=1.00 sl=1155.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1225.00 | 1133.82 | 1133.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 1256.00 | 1158.68 | 1148.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1408.60 | 1417.32 | 1348.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:30:00 | 1412.00 | 1417.32 | 1348.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1350.10 | 1414.10 | 1351.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 1365.80 | 1414.10 | 1351.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1343.10 | 1413.39 | 1351.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1343.10 | 1413.39 | 1351.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1338.30 | 1412.64 | 1351.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 1338.30 | 1412.64 | 1351.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1344.10 | 1411.35 | 1351.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 1344.10 | 1411.35 | 1351.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1345.30 | 1410.70 | 1351.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 1335.50 | 1410.70 | 1351.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1348.00 | 1409.45 | 1351.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1336.00 | 1409.45 | 1351.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1338.60 | 1408.74 | 1350.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1332.80 | 1408.74 | 1350.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1335.60 | 1408.01 | 1350.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 1336.10 | 1408.01 | 1350.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1345.50 | 1402.90 | 1350.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 1343.50 | 1402.90 | 1350.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1350.20 | 1402.37 | 1350.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 1364.20 | 1401.53 | 1350.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-13 09:15:00 | 1500.62 | 1405.53 | 1362.56 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 09:45:00 | 588.03 | 2024-06-04 10:15:00 | 549.88 | STOP_HIT | 1.00 | -6.49% |
| BUY | retest2 | 2024-06-06 14:30:00 | 601.50 | 2024-06-12 13:15:00 | 661.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1097.90 | 2025-06-30 13:15:00 | 1160.70 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-06-27 11:45:00 | 1115.30 | 2025-06-30 13:15:00 | 1160.70 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-07-01 13:15:00 | 1120.20 | 2025-07-11 13:15:00 | 1141.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-07-01 14:45:00 | 1120.00 | 2025-07-11 13:15:00 | 1141.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-07-07 13:45:00 | 1128.40 | 2025-07-11 13:15:00 | 1141.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-07 14:15:00 | 1128.00 | 2025-07-11 13:15:00 | 1141.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-07-07 15:15:00 | 1128.00 | 2025-07-11 13:15:00 | 1141.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-07-09 10:00:00 | 1124.70 | 2025-07-14 09:15:00 | 1143.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-07-11 10:00:00 | 1127.40 | 2025-07-14 10:15:00 | 1157.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-07-14 09:15:00 | 1127.50 | 2025-07-14 10:15:00 | 1157.50 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-10-01 14:00:00 | 1364.20 | 2025-10-13 09:15:00 | 1500.62 | TARGET_HIT | 1.00 | 10.00% |
