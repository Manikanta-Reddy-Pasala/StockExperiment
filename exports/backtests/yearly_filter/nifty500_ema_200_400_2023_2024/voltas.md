# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1323.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 17 |
| ALERT2 | 16 |
| ALERT2_SKIP | 7 |
| ALERT3 | 90 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 3 |
| TARGET_HIT | 10 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 44
- **Target hits / Stop hits / Partials:** 10 / 44 / 3
- **Avg / median % per leg:** 0.56% / -1.56%
- **Sum % (uncompounded):** 32.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 7 | 15.9% | 7 | 37 | 0 | 0.14% | 6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 7 | 15.9% | 7 | 37 | 0 | 0.14% | 6.2% |
| SELL (all) | 13 | 6 | 46.2% | 3 | 7 | 3 | 1.99% | 25.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 3 | 7 | 3 | 1.99% | 25.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 13 | 22.8% | 10 | 44 | 3 | 0.56% | 32.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 11:15:00 | 823.35 | 795.69 | 795.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 10:15:00 | 832.50 | 803.37 | 799.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 09:15:00 | 855.55 | 859.68 | 838.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-22 10:00:00 | 855.55 | 859.68 | 838.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 848.55 | 865.25 | 848.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 855.45 | 865.25 | 848.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 13:45:00 | 853.40 | 864.73 | 848.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 10:15:00 | 841.25 | 863.87 | 849.43 | SL hit (close<static) qty=1.00 sl=848.15 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 14:15:00 | 813.60 | 842.05 | 842.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 12:15:00 | 812.00 | 837.61 | 839.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 846.70 | 834.03 | 837.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 846.70 | 834.03 | 837.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 846.70 | 834.03 | 837.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:00:00 | 846.70 | 834.03 | 837.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 841.90 | 834.11 | 837.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:00:00 | 839.00 | 835.64 | 838.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 12:15:00 | 838.65 | 832.94 | 836.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 13:30:00 | 839.20 | 833.06 | 836.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 10:45:00 | 840.55 | 833.31 | 836.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 840.30 | 833.31 | 836.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:00:00 | 840.30 | 833.31 | 836.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 846.80 | 833.45 | 836.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:45:00 | 847.20 | 833.45 | 836.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 840.90 | 834.08 | 836.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 11:00:00 | 840.90 | 834.08 | 836.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-07 12:15:00 | 860.35 | 834.41 | 836.48 | SL hit (close>static) qty=1.00 sl=847.20 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 858.00 | 838.44 | 838.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 11:15:00 | 862.60 | 841.62 | 840.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 1070.90 | 1072.95 | 1029.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 14:45:00 | 1071.90 | 1072.95 | 1029.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 1034.45 | 1070.69 | 1034.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:00:00 | 1034.45 | 1070.69 | 1034.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 1035.80 | 1070.34 | 1034.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:45:00 | 1042.15 | 1070.34 | 1034.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 13:15:00 | 1033.65 | 1069.97 | 1034.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:00:00 | 1033.65 | 1069.97 | 1034.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 1042.15 | 1069.70 | 1034.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 09:15:00 | 1053.00 | 1069.40 | 1034.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 09:45:00 | 1052.05 | 1069.22 | 1034.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 10:15:00 | 1052.65 | 1069.22 | 1034.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:45:00 | 1052.60 | 1068.02 | 1036.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1040.65 | 1067.41 | 1037.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1040.65 | 1067.41 | 1037.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1037.25 | 1067.11 | 1037.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:45:00 | 1033.05 | 1067.11 | 1037.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1038.80 | 1066.83 | 1037.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 1035.00 | 1066.83 | 1037.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 1043.95 | 1066.60 | 1037.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 13:30:00 | 1045.50 | 1066.39 | 1037.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:15:00 | 1052.85 | 1065.97 | 1037.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 1035.15 | 1065.67 | 1037.25 | SL hit (close<static) qty=1.00 sl=1036.80 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 1659.95 | 1729.12 | 1729.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 12:15:00 | 1645.60 | 1724.12 | 1726.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1721.95 | 1712.06 | 1719.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 1721.95 | 1712.06 | 1719.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1721.95 | 1712.06 | 1719.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1721.95 | 1712.06 | 1719.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1722.00 | 1712.16 | 1719.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1714.55 | 1712.16 | 1719.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 1716.45 | 1705.62 | 1715.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:00:00 | 1716.45 | 1705.62 | 1715.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1711.10 | 1705.67 | 1715.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 1711.85 | 1705.67 | 1715.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1759.80 | 1706.30 | 1715.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 1759.80 | 1706.30 | 1715.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1751.20 | 1706.74 | 1715.88 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 1765.65 | 1724.13 | 1723.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 12:15:00 | 1777.10 | 1724.66 | 1724.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 1736.45 | 1737.56 | 1731.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 1736.45 | 1737.56 | 1731.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1729.20 | 1737.54 | 1731.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 1729.20 | 1737.54 | 1731.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1723.55 | 1737.40 | 1731.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 1723.55 | 1737.40 | 1731.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 1703.00 | 1725.97 | 1726.06 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 1766.05 | 1726.44 | 1726.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 11:15:00 | 1773.80 | 1726.91 | 1726.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1724.55 | 1751.11 | 1740.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1724.55 | 1751.11 | 1740.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1724.55 | 1751.11 | 1740.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 1724.55 | 1751.11 | 1740.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1727.00 | 1750.87 | 1739.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 1721.65 | 1750.87 | 1739.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1748.05 | 1750.33 | 1740.01 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 1634.10 | 1731.22 | 1731.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1621.15 | 1727.39 | 1729.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1363.85 | 1359.65 | 1462.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1363.85 | 1359.65 | 1462.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1438.00 | 1377.52 | 1449.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:30:00 | 1446.65 | 1377.52 | 1449.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1437.00 | 1380.07 | 1449.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 1421.45 | 1380.53 | 1449.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 1415.85 | 1380.53 | 1449.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1469.85 | 1387.09 | 1448.74 | SL hit (close>static) qty=1.00 sl=1463.85 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1367.90 | 1309.03 | 1309.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1372.20 | 1310.22 | 1309.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1344.90 | 1346.05 | 1331.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 1344.40 | 1346.05 | 1331.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1325.90 | 1345.73 | 1331.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1325.90 | 1345.73 | 1331.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1331.00 | 1345.59 | 1331.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:15:00 | 1324.30 | 1345.59 | 1331.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1322.20 | 1345.35 | 1331.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 1322.20 | 1345.35 | 1331.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1312.70 | 1343.93 | 1331.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 1312.70 | 1343.93 | 1331.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1334.70 | 1341.80 | 1330.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 1330.70 | 1341.80 | 1330.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1326.30 | 1341.47 | 1330.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 1335.50 | 1340.90 | 1330.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 1337.10 | 1340.86 | 1330.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 1336.70 | 1340.70 | 1330.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1314.60 | 1340.12 | 1330.82 | SL hit (close<static) qty=1.00 sl=1315.30 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1244.40 | 1323.74 | 1324.08 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1370.50 | 1323.49 | 1323.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1373.30 | 1334.55 | 1329.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 10:15:00 | 1386.30 | 1386.66 | 1364.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 11:00:00 | 1386.30 | 1386.66 | 1364.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1360.90 | 1385.79 | 1365.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1360.90 | 1385.79 | 1365.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1367.40 | 1385.61 | 1365.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1361.80 | 1385.61 | 1365.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1364.80 | 1384.88 | 1365.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 1356.50 | 1384.88 | 1365.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1366.10 | 1384.69 | 1365.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1368.40 | 1384.69 | 1365.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:00:00 | 1370.50 | 1383.87 | 1365.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1349.80 | 1382.95 | 1365.59 | SL hit (close<static) qty=1.00 sl=1362.20 alert=retest2 |

### Cycle 12 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1329.60 | 1373.01 | 1373.21 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 1399.40 | 1373.36 | 1373.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 15:15:00 | 1404.00 | 1374.23 | 1373.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 10:15:00 | 1366.50 | 1378.01 | 1375.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 1366.50 | 1378.01 | 1375.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1366.50 | 1378.01 | 1375.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 1366.50 | 1378.01 | 1375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1360.20 | 1377.83 | 1375.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 1357.50 | 1377.83 | 1375.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1355.00 | 1377.45 | 1375.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1355.00 | 1377.45 | 1375.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1379.40 | 1375.87 | 1374.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1374.70 | 1375.87 | 1374.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1368.90 | 1376.30 | 1375.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1386.70 | 1376.42 | 1375.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1384.10 | 1376.31 | 1375.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1360.40 | 1375.86 | 1374.95 | SL hit (close<static) qty=1.00 sl=1365.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1352.00 | 1373.97 | 1374.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1339.60 | 1373.63 | 1373.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1358.90 | 1358.42 | 1365.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 1363.00 | 1358.42 | 1365.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1365.80 | 1358.56 | 1365.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 1365.80 | 1358.56 | 1365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1367.20 | 1358.65 | 1365.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1367.20 | 1358.65 | 1365.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1377.90 | 1358.84 | 1365.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1377.90 | 1358.84 | 1365.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1375.30 | 1359.25 | 1365.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 1375.30 | 1359.25 | 1365.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1368.90 | 1367.56 | 1368.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1369.30 | 1367.56 | 1368.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1375.20 | 1367.64 | 1368.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1377.40 | 1367.64 | 1368.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1372.60 | 1367.68 | 1368.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1385.70 | 1367.68 | 1368.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1406.80 | 1370.54 | 1370.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1417.70 | 1373.42 | 1372.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1416.10 | 1416.94 | 1398.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:30:00 | 1414.50 | 1416.94 | 1398.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1404.10 | 1416.42 | 1398.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:30:00 | 1400.70 | 1416.42 | 1398.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1399.70 | 1416.26 | 1398.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1399.70 | 1416.26 | 1398.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1398.00 | 1416.08 | 1398.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 1400.20 | 1416.08 | 1398.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1398.00 | 1415.90 | 1398.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1385.20 | 1415.90 | 1398.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1389.50 | 1415.63 | 1398.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1383.00 | 1415.63 | 1398.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1377.60 | 1415.25 | 1398.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1377.60 | 1415.25 | 1398.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 1348.60 | 1383.72 | 1383.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 1340.00 | 1382.90 | 1383.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1406.60 | 1368.99 | 1375.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1406.60 | 1368.99 | 1375.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1406.60 | 1368.99 | 1375.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 1407.90 | 1368.99 | 1375.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1403.50 | 1369.34 | 1375.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:15:00 | 1392.50 | 1369.34 | 1375.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 1418.10 | 1370.88 | 1376.58 | SL hit (close>static) qty=1.00 sl=1413.20 alert=retest2 |

### Cycle 17 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1498.60 | 1382.21 | 1381.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1501.70 | 1384.54 | 1383.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1447.40 | 1479.45 | 1443.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 1447.40 | 1479.45 | 1443.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1447.40 | 1479.45 | 1443.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 1447.40 | 1479.45 | 1443.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1441.50 | 1479.07 | 1443.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:30:00 | 1435.00 | 1479.07 | 1443.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1446.00 | 1478.74 | 1443.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:45:00 | 1438.20 | 1478.74 | 1443.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1446.50 | 1478.42 | 1443.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 1440.60 | 1478.42 | 1443.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1450.60 | 1478.15 | 1443.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:45:00 | 1445.40 | 1478.15 | 1443.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1439.00 | 1477.76 | 1443.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 1439.00 | 1477.76 | 1443.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 1430.20 | 1477.28 | 1443.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 1459.00 | 1477.28 | 1443.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 09:45:00 | 1455.30 | 1476.60 | 1445.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:45:00 | 1441.20 | 1475.37 | 1445.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 1454.20 | 1474.19 | 1444.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1454.50 | 1473.75 | 1444.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1499.30 | 1472.55 | 1445.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 1426.50 | 1472.34 | 1446.02 | SL hit (close<static) qty=1.00 sl=1430.00 alert=retest2 |

### Cycle 18 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 1253.80 | 1426.89 | 1427.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1225.80 | 1393.65 | 1409.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1350.90 | 1344.12 | 1378.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 1350.90 | 1344.12 | 1378.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1402.00 | 1345.34 | 1378.01 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 1507.20 | 1399.88 | 1399.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1510.70 | 1402.01 | 1400.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1404.80 | 1412.92 | 1406.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1404.80 | 1412.92 | 1406.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1404.80 | 1412.92 | 1406.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 1404.80 | 1412.92 | 1406.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1394.80 | 1412.74 | 1406.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 1389.60 | 1412.74 | 1406.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 1324.50 | 1401.18 | 1401.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1323.00 | 1400.40 | 1400.97 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-10 09:15:00 | 855.45 | 2023-10-13 10:15:00 | 841.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-10-10 13:45:00 | 853.40 | 2023-10-13 10:15:00 | 841.25 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-10-16 11:30:00 | 853.60 | 2023-10-17 13:15:00 | 847.90 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-10-16 14:30:00 | 853.90 | 2023-10-17 13:15:00 | 847.90 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-10-18 09:15:00 | 854.20 | 2023-10-18 10:15:00 | 841.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-11-23 12:00:00 | 839.00 | 2023-12-07 12:15:00 | 860.35 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2023-12-04 12:15:00 | 838.65 | 2023-12-07 12:15:00 | 860.35 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2023-12-04 13:30:00 | 839.20 | 2023-12-07 12:15:00 | 860.35 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2023-12-05 10:45:00 | 840.55 | 2023-12-07 12:15:00 | 860.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-03-14 09:15:00 | 1053.00 | 2024-03-20 09:15:00 | 1035.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-03-14 09:45:00 | 1052.05 | 2024-03-20 09:15:00 | 1035.15 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-03-14 10:15:00 | 1052.65 | 2024-04-01 14:15:00 | 1152.63 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2024-03-18 12:45:00 | 1052.60 | 2024-04-01 14:15:00 | 1151.98 | TARGET_HIT | 1.00 | 9.44% |
| BUY | retest2 | 2024-03-19 13:30:00 | 1045.50 | 2024-04-02 09:15:00 | 1158.30 | TARGET_HIT | 1.00 | 10.79% |
| BUY | retest2 | 2024-03-20 09:15:00 | 1052.85 | 2024-04-02 09:15:00 | 1157.26 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2024-03-20 10:45:00 | 1047.85 | 2024-04-02 09:15:00 | 1157.92 | TARGET_HIT | 1.00 | 10.50% |
| BUY | retest2 | 2024-03-20 12:45:00 | 1047.25 | 2024-04-02 09:15:00 | 1157.86 | TARGET_HIT | 1.00 | 10.56% |
| BUY | retest2 | 2024-08-05 12:15:00 | 1448.70 | 2024-08-08 15:15:00 | 1425.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-08-05 13:00:00 | 1447.10 | 2024-08-08 15:15:00 | 1425.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-08 13:30:00 | 1452.75 | 2024-08-08 15:15:00 | 1425.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-08-12 09:15:00 | 1516.80 | 2024-08-20 13:15:00 | 1668.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-08 09:15:00 | 1778.00 | 2024-11-11 13:15:00 | 1746.25 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-11-11 09:30:00 | 1779.40 | 2024-11-11 13:15:00 | 1746.25 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-11-11 10:30:00 | 1774.90 | 2024-11-11 13:15:00 | 1746.25 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-03-13 10:45:00 | 1421.45 | 2025-03-18 09:15:00 | 1469.85 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-03-13 11:15:00 | 1415.85 | 2025-03-18 09:15:00 | 1469.85 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-03-21 13:30:00 | 1422.00 | 2025-04-01 14:15:00 | 1350.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 13:15:00 | 1421.15 | 2025-04-01 14:15:00 | 1350.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1384.55 | 2025-04-04 09:15:00 | 1315.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-21 13:30:00 | 1422.00 | 2025-04-07 09:15:00 | 1279.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 13:15:00 | 1421.15 | 2025-04-07 09:15:00 | 1279.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1384.55 | 2025-04-07 09:15:00 | 1246.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-30 14:15:00 | 1335.50 | 2025-08-01 09:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-30 15:00:00 | 1337.10 | 2025-08-01 09:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-31 10:15:00 | 1336.70 | 2025-08-01 09:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-08-04 13:45:00 | 1336.00 | 2025-08-05 15:15:00 | 1314.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-24 11:15:00 | 1368.40 | 2025-09-26 09:15:00 | 1349.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-25 10:00:00 | 1370.50 | 2025-09-26 09:15:00 | 1349.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-09-29 14:00:00 | 1373.80 | 2025-09-30 10:15:00 | 1359.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-09-30 12:15:00 | 1370.00 | 2025-09-30 13:15:00 | 1359.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-01 09:15:00 | 1356.00 | 2025-10-06 11:15:00 | 1334.20 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-10-01 11:00:00 | 1355.20 | 2025-10-06 11:15:00 | 1334.20 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-01 14:45:00 | 1355.40 | 2025-10-06 11:15:00 | 1334.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-10-01 15:15:00 | 1354.30 | 2025-10-06 11:15:00 | 1334.20 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-09 09:45:00 | 1391.80 | 2025-11-03 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-10-13 11:30:00 | 1375.00 | 2025-11-03 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-13 12:15:00 | 1374.80 | 2025-11-03 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-10-13 12:45:00 | 1376.20 | 2025-11-03 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1386.70 | 2025-12-01 14:15:00 | 1360.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1384.10 | 2025-12-01 14:15:00 | 1360.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-04 11:15:00 | 1392.50 | 2026-02-04 14:15:00 | 1418.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-03-05 09:15:00 | 1459.00 | 2026-03-12 09:15:00 | 1426.50 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-09 09:45:00 | 1455.30 | 2026-03-12 09:15:00 | 1426.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-03-09 12:45:00 | 1441.20 | 2026-03-12 09:15:00 | 1426.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-03-10 09:15:00 | 1454.20 | 2026-03-12 09:15:00 | 1426.50 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-03-11 09:15:00 | 1499.30 | 2026-03-12 09:15:00 | 1426.50 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2026-03-12 13:30:00 | 1470.70 | 2026-03-13 09:15:00 | 1396.80 | STOP_HIT | 1.00 | -5.02% |
