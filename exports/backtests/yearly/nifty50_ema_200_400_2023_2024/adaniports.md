# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2022-04-07 09:15:00 → 2026-05-08 15:15:00 (7054 bars)
- **Last close:** 1760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 44 |
| PARTIAL | 0 |
| TARGET_HIT | 15 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 30
- **Target hits / Stop hits / Partials:** 15 / 30 / 0
- **Avg / median % per leg:** 1.86% / -1.16%
- **Sum % (uncompounded):** 83.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 15 | 41.7% | 15 | 21 | 0 | 2.92% | 105.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.32% | -5.3% |
| BUY @ 3rd Alert (retest2) | 35 | 15 | 42.9% | 15 | 20 | 0 | 3.15% | 110.4% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.38% | -21.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.38% | -21.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.32% | -5.3% |
| retest2 (combined) | 44 | 15 | 34.1% | 15 | 29 | 0 | 2.02% | 89.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-12 10:15:00 | 701.45 | 670.14 | 670.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 732.40 | 675.44 | 673.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 719.75 | 727.19 | 710.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-23 10:00:00 | 719.75 | 727.19 | 710.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 711.65 | 727.03 | 710.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 11:00:00 | 711.65 | 727.03 | 710.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 717.95 | 726.36 | 710.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 11:45:00 | 724.85 | 726.26 | 710.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 12:15:00 | 723.00 | 726.26 | 710.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 13:30:00 | 723.25 | 726.18 | 710.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 14:00:00 | 723.30 | 726.18 | 710.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 717.50 | 730.75 | 716.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 13:30:00 | 717.05 | 730.75 | 716.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 709.75 | 730.31 | 716.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:00:00 | 709.75 | 730.31 | 716.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 717.00 | 730.18 | 716.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 13:30:00 | 719.85 | 729.83 | 716.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 15:15:00 | 720.50 | 729.72 | 716.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 14:15:00 | 719.65 | 729.04 | 717.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 11:30:00 | 719.40 | 728.54 | 717.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-07 14:15:00 | 791.84 | 747.01 | 733.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 1410.90 | 1463.23 | 1463.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1385.90 | 1454.80 | 1458.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1411.90 | 1399.79 | 1423.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 11:00:00 | 1411.90 | 1399.79 | 1423.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1142.10 | 1101.70 | 1139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:00:00 | 1142.10 | 1101.70 | 1139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1147.25 | 1102.15 | 1139.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 1147.25 | 1102.15 | 1139.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 1136.80 | 1102.50 | 1139.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-06 15:15:00 | 1130.80 | 1102.50 | 1139.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 09:15:00 | 1151.00 | 1103.26 | 1139.55 | SL hit (close>static) qty=1.00 sl=1147.95 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 1201.30 | 1155.00 | 1154.82 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1085.85 | 1154.57 | 1154.61 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 1214.90 | 1154.39 | 1154.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.17 | 1155.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.66 | 1333.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1411.80 | 1397.89 | 1335.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | SL hit (close<ema400) qty=1.00 sl=1338.55 alert=retest1 |

### Cycle 6 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.66 | 1376.84 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1407.20 | 1368.43 | 1368.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.21 | 1368.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1388.00 | 1391.40 | 1381.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 1380.00 | 1391.29 | 1381.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.29 | 1381.46 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.11 | 1465.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1396.30 | 1463.83 | 1464.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1568.50 | 1457.81 | 1456.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.47 | 1461.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 1598.10 | 1450.98 | 1449.90 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-26 11:45:00 | 724.85 | 2023-08-07 14:15:00 | 791.84 | TARGET_HIT | 1.00 | 9.24% |
| BUY | retest2 | 2023-06-26 12:15:00 | 723.00 | 2023-08-07 14:15:00 | 792.55 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2023-06-26 13:30:00 | 723.25 | 2023-08-07 14:15:00 | 791.62 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2023-06-26 14:00:00 | 723.30 | 2023-08-07 14:15:00 | 791.34 | TARGET_HIT | 1.00 | 9.41% |
| BUY | retest2 | 2023-07-10 13:30:00 | 719.85 | 2023-08-08 14:15:00 | 797.34 | TARGET_HIT | 1.00 | 10.76% |
| BUY | retest2 | 2023-07-10 15:15:00 | 720.50 | 2023-08-08 14:15:00 | 795.30 | TARGET_HIT | 1.00 | 10.38% |
| BUY | retest2 | 2023-07-13 14:15:00 | 719.65 | 2023-08-08 14:15:00 | 795.58 | TARGET_HIT | 1.00 | 10.55% |
| BUY | retest2 | 2023-07-14 11:30:00 | 719.40 | 2023-08-08 14:15:00 | 795.63 | TARGET_HIT | 1.00 | 10.60% |
| BUY | retest2 | 2023-10-10 14:00:00 | 821.60 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-10-11 09:15:00 | 823.35 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-10-11 15:15:00 | 819.00 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2023-10-13 13:45:00 | 819.30 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2023-10-17 09:15:00 | 811.15 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-10-18 09:30:00 | 807.40 | 2023-10-18 10:15:00 | 800.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-11-08 09:45:00 | 808.35 | 2023-11-20 15:15:00 | 802.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-11-09 15:00:00 | 806.75 | 2023-11-20 15:15:00 | 802.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-11-21 09:15:00 | 811.00 | 2023-11-21 15:15:00 | 801.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-11-21 12:30:00 | 807.45 | 2023-11-21 15:15:00 | 801.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-11-21 13:15:00 | 807.25 | 2023-11-21 15:15:00 | 801.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-11-28 09:15:00 | 819.55 | 2023-12-05 09:15:00 | 901.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 10:15:00 | 1248.50 | 2024-04-01 10:15:00 | 1373.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-18 09:45:00 | 1246.80 | 2024-04-01 10:15:00 | 1371.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-18 12:00:00 | 1249.25 | 2024-04-01 10:15:00 | 1374.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-19 11:45:00 | 1245.75 | 2024-04-01 10:15:00 | 1370.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 12:30:00 | 1308.90 | 2024-05-06 09:15:00 | 1265.15 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-04-19 13:45:00 | 1312.05 | 2024-05-06 09:15:00 | 1265.15 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2024-05-13 14:30:00 | 1309.50 | 2024-05-23 14:15:00 | 1440.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 10:30:00 | 1307.35 | 2024-05-23 14:15:00 | 1438.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-12 10:45:00 | 1521.00 | 2024-08-14 10:15:00 | 1455.20 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2024-08-12 14:30:00 | 1513.05 | 2024-08-14 10:15:00 | 1455.20 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-08-21 10:15:00 | 1514.55 | 2024-08-29 11:15:00 | 1456.95 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-03-06 15:15:00 | 1130.80 | 2025-03-07 09:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-03-11 09:30:00 | 1133.70 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-11 10:15:00 | 1134.85 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-03-11 15:00:00 | 1136.10 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-12 11:15:00 | 1119.35 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-03-12 11:45:00 | 1113.30 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1117.35 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1119.55 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-03-17 12:30:00 | 1128.25 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1411.80 | 2025-06-19 12:15:00 | 1336.70 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1345.50 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-20 13:15:00 | 1343.40 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-20 14:45:00 | 1344.30 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-23 09:45:00 | 1344.00 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.14% |
