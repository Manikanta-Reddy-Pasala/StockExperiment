# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 2748.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 44
- **Target hits / Stop hits / Partials:** 6 / 48 / 11
- **Avg / median % per leg:** 0.03% / -1.70%
- **Sum % (uncompounded):** 1.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 2 | 8 | 0 | -0.61% | -6.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 2 | 8 | 0 | -0.61% | -6.1% |
| SELL (all) | 55 | 19 | 34.5% | 4 | 40 | 11 | 0.14% | 7.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 19 | 34.5% | 4 | 40 | 11 | 0.14% | 7.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 21 | 32.3% | 6 | 48 | 11 | 0.03% | 1.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 606.70 | 622.53 | 622.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 11:15:00 | 600.30 | 622.13 | 622.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 12:15:00 | 597.50 | 595.50 | 606.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-08 13:00:00 | 597.50 | 595.50 | 606.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 577.78 | 560.06 | 574.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 10:45:00 | 573.53 | 560.18 | 574.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 13:00:00 | 572.95 | 560.42 | 574.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:45:00 | 573.90 | 560.99 | 574.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 14:45:00 | 573.45 | 562.90 | 574.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 573.50 | 563.01 | 574.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:15:00 | 575.60 | 563.01 | 574.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 574.28 | 563.12 | 574.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:30:00 | 576.48 | 563.12 | 574.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 572.78 | 563.22 | 574.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:30:00 | 574.28 | 563.22 | 574.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 544.85 | 563.07 | 574.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 544.30 | 563.07 | 574.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 545.20 | 563.07 | 574.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 544.78 | 563.07 | 574.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-12-21 09:15:00 | 516.18 | 562.36 | 574.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 575.50 | 560.68 | 560.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 584.28 | 560.91 | 560.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 586.50 | 593.83 | 582.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:30:00 | 586.03 | 593.83 | 582.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 575.80 | 593.60 | 582.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 575.80 | 593.60 | 582.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 583.92 | 593.50 | 582.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 598.50 | 593.24 | 582.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 588.03 | 605.63 | 595.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 549.88 | 605.07 | 595.49 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-13 12:15:00)

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

### Cycle 4 — BUY (started 2025-07-16 15:15:00)

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
| BUY | retest2 | 2023-07-17 09:30:00 | 597.25 | 2023-07-17 14:15:00 | 587.13 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-07-17 10:00:00 | 596.55 | 2023-07-17 14:15:00 | 587.13 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-07-17 10:45:00 | 595.17 | 2023-07-17 14:15:00 | 587.13 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-07-17 11:45:00 | 595.35 | 2023-07-17 14:15:00 | 587.13 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-09-22 11:45:00 | 639.38 | 2023-09-26 10:15:00 | 622.78 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2023-09-25 09:45:00 | 641.00 | 2023-09-26 10:15:00 | 622.78 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2023-12-15 10:45:00 | 573.53 | 2023-12-20 13:15:00 | 544.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-15 13:00:00 | 572.95 | 2023-12-20 13:15:00 | 544.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-18 10:45:00 | 573.90 | 2023-12-20 13:15:00 | 545.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-19 14:45:00 | 573.45 | 2023-12-20 13:15:00 | 544.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-15 10:45:00 | 573.53 | 2023-12-21 09:15:00 | 516.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-15 13:00:00 | 572.95 | 2023-12-21 09:15:00 | 515.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-18 10:45:00 | 573.90 | 2023-12-21 09:15:00 | 516.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-19 14:45:00 | 573.45 | 2023-12-21 09:15:00 | 516.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-22 12:30:00 | 586.90 | 2023-12-29 14:15:00 | 561.43 | PARTIAL | 0.50 | 4.34% |
| SELL | retest2 | 2023-12-22 13:45:00 | 590.98 | 2023-12-29 14:15:00 | 560.74 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2023-12-26 12:15:00 | 590.25 | 2023-12-29 14:15:00 | 560.15 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2023-12-27 12:15:00 | 589.63 | 2023-12-29 15:15:00 | 557.55 | PARTIAL | 0.50 | 5.44% |
| SELL | retest2 | 2023-12-22 12:30:00 | 586.90 | 2024-01-01 09:15:00 | 568.85 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2023-12-22 13:45:00 | 590.98 | 2024-01-01 09:15:00 | 568.85 | STOP_HIT | 0.50 | 3.74% |
| SELL | retest2 | 2023-12-26 12:15:00 | 590.25 | 2024-01-01 09:15:00 | 568.85 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2023-12-27 12:15:00 | 589.63 | 2024-01-01 09:15:00 | 568.85 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2024-01-01 14:15:00 | 572.85 | 2024-01-03 13:15:00 | 584.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-01-01 15:00:00 | 572.78 | 2024-01-03 13:15:00 | 584.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-01-02 10:15:00 | 567.20 | 2024-01-03 13:15:00 | 584.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-01-03 10:30:00 | 573.20 | 2024-01-03 13:15:00 | 584.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-01-05 15:15:00 | 574.00 | 2024-01-11 11:15:00 | 583.95 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-01-08 12:30:00 | 572.92 | 2024-01-11 11:15:00 | 583.95 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-01-09 09:30:00 | 573.63 | 2024-01-11 11:15:00 | 583.95 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-01-09 13:15:00 | 573.80 | 2024-01-11 11:15:00 | 583.95 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-01-16 13:00:00 | 560.35 | 2024-01-29 11:15:00 | 584.05 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-01-17 13:45:00 | 564.92 | 2024-01-29 11:15:00 | 584.05 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-01-17 15:00:00 | 565.05 | 2024-01-29 11:15:00 | 584.05 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-01-18 09:15:00 | 563.48 | 2024-01-29 11:15:00 | 584.05 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-02-09 09:45:00 | 543.88 | 2024-03-04 09:15:00 | 581.25 | STOP_HIT | 1.00 | -6.87% |
| SELL | retest2 | 2024-02-09 10:15:00 | 542.67 | 2024-03-04 09:15:00 | 581.25 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2024-02-09 14:00:00 | 543.92 | 2024-03-04 09:15:00 | 581.25 | STOP_HIT | 1.00 | -6.86% |
| SELL | retest2 | 2024-02-12 09:45:00 | 543.30 | 2024-03-04 09:15:00 | 581.25 | STOP_HIT | 1.00 | -6.99% |
| SELL | retest2 | 2024-03-11 10:00:00 | 545.50 | 2024-03-13 09:15:00 | 518.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 10:45:00 | 547.50 | 2024-03-13 09:15:00 | 520.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 12:00:00 | 547.10 | 2024-03-13 09:15:00 | 519.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 10:00:00 | 545.50 | 2024-03-18 12:15:00 | 551.00 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2024-03-11 10:45:00 | 547.50 | 2024-03-18 12:15:00 | 551.00 | STOP_HIT | 0.50 | -0.64% |
| SELL | retest2 | 2024-03-11 12:00:00 | 547.10 | 2024-03-18 12:15:00 | 551.00 | STOP_HIT | 0.50 | -0.71% |
| SELL | retest2 | 2024-03-18 10:00:00 | 545.03 | 2024-03-19 09:15:00 | 564.35 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-03-20 10:30:00 | 553.53 | 2024-03-22 14:15:00 | 565.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-03-21 13:45:00 | 550.95 | 2024-03-22 14:15:00 | 565.05 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-03-27 15:00:00 | 553.33 | 2024-04-01 09:15:00 | 565.05 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-03-28 12:30:00 | 552.95 | 2024-04-01 09:15:00 | 565.05 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-04-01 13:15:00 | 559.35 | 2024-04-02 09:15:00 | 567.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-04-01 15:00:00 | 555.28 | 2024-04-02 09:15:00 | 567.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-05-08 09:15:00 | 598.50 | 2024-06-04 10:15:00 | 549.88 | STOP_HIT | 1.00 | -8.12% |
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
