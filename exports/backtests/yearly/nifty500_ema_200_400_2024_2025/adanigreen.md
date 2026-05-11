# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 12 |
| TARGET_HIT | 13 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 41
- **Target hits / Stop hits / Partials:** 13 / 43 / 12
- **Avg / median % per leg:** 1.84% / -0.67%
- **Sum % (uncompounded):** 124.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 6 | 21.4% | 6 | 22 | 0 | 0.88% | 24.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 6 | 21.4% | 6 | 22 | 0 | 0.88% | 24.6% |
| SELL (all) | 40 | 21 | 52.5% | 7 | 21 | 12 | 2.51% | 100.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 21 | 52.5% | 7 | 21 | 12 | 2.51% | 100.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 27 | 39.7% | 13 | 43 | 12 | 1.84% | 125.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1781.20 | 1816.84 | 1816.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 1771.00 | 1813.80 | 1815.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1788.00 | 1774.66 | 1791.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:00:00 | 1788.00 | 1774.66 | 1791.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1770.10 | 1774.62 | 1791.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:45:00 | 1765.00 | 1774.62 | 1791.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1821.15 | 1768.55 | 1786.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1821.15 | 1768.55 | 1786.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1848.00 | 1769.34 | 1786.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1841.05 | 1769.34 | 1786.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1828.00 | 1773.68 | 1788.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:45:00 | 1827.00 | 1773.68 | 1788.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1794.25 | 1799.76 | 1800.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:45:00 | 1788.60 | 1799.72 | 1800.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 12:30:00 | 1782.50 | 1799.53 | 1800.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 14:00:00 | 1788.95 | 1799.42 | 1799.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 1786.75 | 1798.81 | 1799.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1775.10 | 1794.28 | 1797.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 1783.95 | 1794.28 | 1797.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1788.00 | 1794.09 | 1797.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1780.30 | 1794.09 | 1797.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1781.05 | 1793.96 | 1797.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:45:00 | 1780.00 | 1793.83 | 1796.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 1780.00 | 1793.69 | 1796.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1699.17 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1693.38 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1699.50 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1697.41 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.28 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1692.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1786.70 | 1791.42 | 1795.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 1800.75 | 1791.42 | 1795.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1798.65 | 1791.49 | 1795.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 1798.65 | 1791.49 | 1795.58 | SL hit (close>ema200) qty=0.50 sl=1791.49 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 1873.50 | 1799.57 | 1799.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1901.20 | 1800.59 | 1799.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 1834.25 | 1836.68 | 1820.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 11:00:00 | 1834.25 | 1836.68 | 1820.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1819.10 | 1836.50 | 1820.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1819.10 | 1836.50 | 1820.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1821.00 | 1836.35 | 1820.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 15:00:00 | 1829.85 | 1836.08 | 1820.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1831.00 | 1836.00 | 1820.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 1834.10 | 1835.98 | 1820.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 1828.10 | 1835.93 | 1820.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1872.85 | 1854.97 | 1834.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1851.00 | 1854.97 | 1834.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1841.00 | 1855.96 | 1836.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 1841.00 | 1855.96 | 1836.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1812.50 | 1855.49 | 1836.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 1812.50 | 1855.49 | 1836.28 | SL hit (close<static) qty=1.00 sl=1817.65 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 1746.60 | 1857.41 | 1857.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1742.55 | 1852.12 | 1854.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 888.45 | 882.61 | 978.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:30:00 | 884.60 | 882.61 | 978.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 966.80 | 889.74 | 966.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:45:00 | 982.55 | 889.74 | 966.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 964.70 | 890.48 | 966.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:45:00 | 950.40 | 891.83 | 966.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 11:00:00 | 951.00 | 894.33 | 966.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 948.20 | 897.34 | 966.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 950.30 | 897.87 | 966.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 958.50 | 903.61 | 963.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 958.50 | 903.61 | 963.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 954.95 | 904.13 | 962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 954.90 | 904.13 | 962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 966.40 | 904.75 | 962.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 966.40 | 904.75 | 962.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 962.70 | 905.32 | 962.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 961.50 | 905.32 | 962.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 957.45 | 905.90 | 962.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 913.42 | 909.21 | 961.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 916.35 | 909.21 | 961.39 | SL hit (close>static) qty=0.50 sl=909.21 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1002.25 | 953.13 | 952.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 1017.90 | 953.77 | 953.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 989.00 | 995.99 | 979.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 989.00 | 995.99 | 979.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 977.50 | 995.67 | 979.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 974.60 | 995.67 | 979.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 989.60 | 995.61 | 979.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 996.90 | 995.53 | 979.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 15:15:00 | 977.50 | 994.64 | 979.95 | SL hit (close<static) qty=1.00 sl=977.90 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 910.05 | 991.45 | 991.65 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1148.65 | 975.00 | 974.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 1158.00 | 976.82 | 975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1030.03 | 1010.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 1027.80 | 1030.03 | 1010.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.45 | 1017.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1019.70 | 1034.45 | 1017.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1017.20 | 1034.02 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1017.20 | 1034.02 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1017.90 | 1033.86 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1015.70 | 1033.86 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.69 | 1017.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1020.50 | 1033.69 | 1017.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1015.40 | 1033.51 | 1017.10 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1050.20 | 1034.11 | 1034.09 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1034.03 | 1034.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1033.91 | 1034.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1034.70 | 1024.53 | 1028.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1027.40 | 1024.56 | 1028.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1022.70 | 1025.96 | 1028.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 971.57 | 1022.88 | 1027.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 10:15:00 | 920.43 | 997.37 | 1012.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-24 09:15:00 | 1927.60 | 2024-06-03 09:15:00 | 2120.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-24 10:15:00 | 1932.15 | 2024-06-03 09:15:00 | 2125.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-27 10:30:00 | 1919.30 | 2024-06-03 09:15:00 | 2111.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-31 09:30:00 | 1930.80 | 2024-06-03 09:15:00 | 2123.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1888.55 | 2024-07-02 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1833.85 | 2024-07-02 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-08-05 11:45:00 | 1788.60 | 2024-08-12 09:15:00 | 1699.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 12:30:00 | 1782.50 | 2024-08-12 09:15:00 | 1693.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 14:00:00 | 1788.95 | 2024-08-12 09:15:00 | 1699.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1786.75 | 2024-08-12 09:15:00 | 1697.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1780.30 | 2024-08-12 09:15:00 | 1691.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1781.05 | 2024-08-12 09:15:00 | 1692.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:45:00 | 1780.00 | 2024-08-12 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 13:00:00 | 1780.00 | 2024-08-12 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 11:45:00 | 1788.60 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.56% |
| SELL | retest2 | 2024-08-05 12:30:00 | 1782.50 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2024-08-05 14:00:00 | 1788.95 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.54% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1786.75 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.67% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1780.30 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1781.05 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.99% |
| SELL | retest2 | 2024-08-09 11:45:00 | 1780.00 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.05% |
| SELL | retest2 | 2024-08-09 13:00:00 | 1780.00 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.05% |
| SELL | retest2 | 2024-08-13 15:15:00 | 1810.00 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-08-14 09:45:00 | 1808.95 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-08-16 09:30:00 | 1810.45 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-08-29 15:00:00 | 1829.85 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-30 10:00:00 | 1831.00 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-30 11:00:00 | 1834.10 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-30 11:30:00 | 1828.10 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1828.90 | 2024-09-12 14:15:00 | 1811.05 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1860.35 | 2024-09-24 09:15:00 | 2046.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-21 13:45:00 | 950.40 | 2025-04-02 09:15:00 | 913.42 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2025-03-21 13:45:00 | 950.40 | 2025-04-02 09:15:00 | 916.35 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-03-24 11:00:00 | 951.00 | 2025-04-07 09:15:00 | 855.36 | TARGET_HIT | 1.00 | 10.06% |
| SELL | retest2 | 2025-03-25 09:15:00 | 948.20 | 2025-04-07 09:15:00 | 855.90 | TARGET_HIT | 1.00 | 9.73% |
| SELL | retest2 | 2025-03-25 10:00:00 | 950.30 | 2025-04-07 09:15:00 | 853.38 | TARGET_HIT | 1.00 | 10.20% |
| SELL | retest2 | 2025-03-28 11:15:00 | 961.50 | 2025-04-07 09:15:00 | 855.27 | TARGET_HIT | 1.00 | 11.05% |
| SELL | retest2 | 2025-03-28 12:15:00 | 957.45 | 2025-04-07 09:15:00 | 861.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 12:30:00 | 959.50 | 2025-04-07 09:15:00 | 863.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-17 10:45:00 | 956.40 | 2025-04-23 14:15:00 | 953.15 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-04-23 10:00:00 | 930.65 | 2025-04-25 10:15:00 | 908.58 | PARTIAL | 0.50 | 2.37% |
| SELL | retest2 | 2025-04-23 10:00:00 | 930.65 | 2025-04-28 09:15:00 | 932.00 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-04-25 09:45:00 | 925.00 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-04-28 12:45:00 | 938.10 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-29 09:30:00 | 937.75 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-05-06 11:15:00 | 934.45 | 2025-05-08 14:15:00 | 887.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 934.45 | 2025-05-12 09:15:00 | 934.95 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2025-05-12 10:00:00 | 934.95 | 2025-05-13 09:15:00 | 956.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-12 11:15:00 | 936.15 | 2025-05-13 09:15:00 | 956.80 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-16 14:00:00 | 996.90 | 2025-06-17 15:15:00 | 977.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-24 11:30:00 | 998.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1003.90 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-04 14:45:00 | 997.30 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-08 13:15:00 | 987.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-08 14:00:00 | 988.50 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-28 09:15:00 | 989.50 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-01 09:15:00 | 994.80 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1020.50 | 2025-10-28 09:15:00 | 1015.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1025.90 | 2025-10-29 11:15:00 | 1128.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:00:00 | 1026.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-24 12:00:00 | 1020.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-27 13:30:00 | 1033.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1033.00 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1035.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1036.80 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-09 09:15:00 | 971.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-19 10:15:00 | 920.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 11:30:00 | 1019.80 | 2026-04-08 15:15:00 | 1035.00 | STOP_HIT | 1.00 | -1.49% |
