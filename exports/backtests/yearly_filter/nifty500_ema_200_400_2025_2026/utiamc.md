# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 973.15
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
| ALERT2_SKIP | 0 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 1.76% / 1.64%
- **Sum % (uncompounded):** 19.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.26% | -9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.26% | -9.1% |
| SELL (all) | 7 | 6 | 85.7% | 1 | 3 | 3 | 4.06% | 28.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 1 | 3 | 3 | 4.06% | 28.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 6 | 54.5% | 1 | 7 | 3 | 1.76% | 19.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1175.40 | 1055.16 | 1054.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 12:15:00 | 1189.00 | 1057.64 | 1056.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 1341.40 | 1350.25 | 1275.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:45:00 | 1342.10 | 1350.25 | 1275.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1312.40 | 1348.57 | 1312.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 1312.90 | 1348.57 | 1312.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1304.30 | 1348.13 | 1312.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 1304.30 | 1348.13 | 1312.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1294.30 | 1340.25 | 1311.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1294.30 | 1340.25 | 1311.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1332.10 | 1353.39 | 1332.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 1331.40 | 1353.39 | 1332.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 1330.00 | 1353.16 | 1332.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 1330.00 | 1353.16 | 1332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1314.90 | 1352.78 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 1314.90 | 1352.78 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1322.00 | 1344.74 | 1330.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 1309.80 | 1344.74 | 1330.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1302.00 | 1343.98 | 1329.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1317.60 | 1343.98 | 1329.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1326.70 | 1338.62 | 1328.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1344.60 | 1330.59 | 1325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1300.00 | 1343.83 | 1333.30 | SL hit (close<static) qty=1.00 sl=1321.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 1243.50 | 1325.78 | 1325.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1236.00 | 1319.56 | 1322.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 1140.50 | 1139.29 | 1176.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:45:00 | 1141.90 | 1139.29 | 1176.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1106.40 | 1058.80 | 1099.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 1083.45 | 1061.09 | 1099.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 1091.70 | 1061.43 | 1099.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 1091.55 | 1062.52 | 1099.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1037.12 | 1065.28 | 1094.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1036.97 | 1065.28 | 1094.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 1065.70 | 1065.28 | 1094.03 | SL hit (close>ema200) qty=0.50 sl=1065.28 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 09:15:00 | 1042.60 | 2025-05-13 10:15:00 | 1056.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-13 09:15:00 | 1344.60 | 2025-10-20 09:15:00 | 1300.00 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1334.50 | 2025-10-23 09:15:00 | 1308.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-10-20 13:00:00 | 1334.90 | 2025-10-23 09:15:00 | 1308.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-24 14:30:00 | 1341.30 | 2025-10-27 11:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1083.45 | 2026-02-20 09:15:00 | 1037.12 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2026-02-12 10:15:00 | 1091.70 | 2026-02-20 09:15:00 | 1036.97 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1083.45 | 2026-02-20 11:15:00 | 1065.70 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-02-12 10:15:00 | 1091.70 | 2026-02-20 11:15:00 | 1065.70 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2026-02-12 13:45:00 | 1091.55 | 2026-02-26 12:15:00 | 1029.28 | PARTIAL | 0.50 | 5.70% |
| SELL | retest2 | 2026-02-12 13:45:00 | 1091.55 | 2026-03-02 09:15:00 | 975.11 | TARGET_HIT | 0.50 | 10.67% |
