# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1268.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 16
- **Target hits / Stop hits / Partials:** 0 / 17 / 1
- **Avg / median % per leg:** -2.39% / -3.24%
- **Sum % (uncompounded):** -43.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.75% | -22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.75% | -22.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -2.11% | -21.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -2.11% | -21.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 2 | 11.1% | 0 | 17 | 1 | -2.39% | -43.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 1256.60 | 1083.58 | 1082.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 1274.30 | 1089.03 | 1085.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1525.10 | 1525.92 | 1435.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 1525.10 | 1525.92 | 1435.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1441.40 | 1518.48 | 1440.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1441.40 | 1518.48 | 1440.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1454.80 | 1517.84 | 1440.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:45:00 | 1457.60 | 1517.19 | 1440.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1429.50 | 1513.48 | 1440.56 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1309.90 | 1454.30 | 1454.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1298.10 | 1452.74 | 1453.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 1039.40 | 1008.26 | 1086.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 1039.40 | 1008.26 | 1086.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1074.80 | 1014.88 | 1074.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1074.80 | 1014.88 | 1074.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1088.40 | 1015.61 | 1075.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 1088.40 | 1015.61 | 1075.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1089.10 | 1016.34 | 1075.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 1089.10 | 1016.34 | 1075.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1084.70 | 1032.45 | 1077.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 1084.70 | 1032.45 | 1077.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1085.00 | 1032.97 | 1078.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 1081.95 | 1034.49 | 1078.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 1027.85 | 1035.07 | 1077.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 1041.05 | 1035.07 | 1077.55 | SL hit (close>static) qty=0.50 sl=1035.07 alert=retest2 |

### Cycle 3 — BUY (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 09:15:00 | 1117.80 | 1100.19 | 1100.18 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1033.70 | 1100.18 | 1100.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1028.00 | 1099.46 | 1099.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1092.90 | 1092.08 | 1095.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 11:00:00 | 1092.90 | 1092.08 | 1095.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1075.60 | 1072.01 | 1083.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1065.00 | 1072.33 | 1083.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:00:00 | 1064.45 | 1072.36 | 1083.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1115.50 | 1072.85 | 1083.50 | SL hit (close>static) qty=1.00 sl=1103.30 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 1205.00 | 1092.67 | 1092.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1219.00 | 1093.93 | 1093.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-25 12:45:00 | 1457.60 | 2025-07-28 10:15:00 | 1429.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-29 15:00:00 | 1462.80 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1458.90 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-30 12:30:00 | 1459.40 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-31 13:00:00 | 1477.00 | 2025-08-01 14:15:00 | 1419.70 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-08-14 11:00:00 | 1473.90 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-18 10:30:00 | 1475.50 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-08-18 11:15:00 | 1474.80 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-02-12 15:00:00 | 1081.95 | 2026-02-13 11:15:00 | 1027.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 15:00:00 | 1081.95 | 2026-02-13 11:15:00 | 1041.05 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1065.70 | 2026-03-10 09:15:00 | 1100.20 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-03-12 09:30:00 | 1078.10 | 2026-03-12 11:15:00 | 1115.20 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-03-12 10:15:00 | 1081.30 | 2026-03-12 11:15:00 | 1115.20 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1086.10 | 2026-03-17 09:15:00 | 1131.20 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2026-03-13 11:45:00 | 1093.20 | 2026-03-17 09:15:00 | 1131.20 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-03-17 15:00:00 | 1093.20 | 2026-03-18 10:15:00 | 1124.60 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1065.00 | 2026-04-10 09:15:00 | 1115.50 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2026-04-09 13:00:00 | 1064.45 | 2026-04-10 09:15:00 | 1115.50 | STOP_HIT | 1.00 | -4.80% |
