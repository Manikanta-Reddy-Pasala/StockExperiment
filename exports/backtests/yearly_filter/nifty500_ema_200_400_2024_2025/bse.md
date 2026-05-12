# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3905.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 0 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 2 |
| TARGET_HIT | 10 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 18
- **Target hits / Stop hits / Partials:** 10 / 21 / 2
- **Avg / median % per leg:** 1.99% / -1.10%
- **Sum % (uncompounded):** 65.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 11 | 42.3% | 9 | 17 | 0 | 1.97% | 51.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 11 | 42.3% | 9 | 17 | 0 | 1.97% | 51.2% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.07% | 14.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.07% | 14.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 15 | 45.5% | 10 | 21 | 2 | 1.99% | 65.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 09:15:00 | 825.67 | 876.86 | 876.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 823.32 | 876.32 | 876.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 09:15:00 | 814.67 | 813.30 | 837.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:00:00 | 814.67 | 813.30 | 837.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 862.67 | 814.25 | 835.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 862.67 | 814.25 | 835.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 861.05 | 814.72 | 835.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 15:00:00 | 851.50 | 816.51 | 835.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 868.68 | 817.39 | 835.94 | SL hit (close>static) qty=1.00 sl=868.67 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 12:15:00 | 896.38 | 844.75 | 844.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 902.00 | 846.81 | 845.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 1481.62 | 1488.19 | 1363.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:00:00 | 1481.62 | 1488.19 | 1363.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 1724.25 | 1837.08 | 1727.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 1724.25 | 1837.08 | 1727.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1760.67 | 1836.32 | 1727.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 13:00:00 | 1765.00 | 1835.61 | 1728.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 1720.00 | 1833.60 | 1728.06 | SL hit (close<static) qty=1.00 sl=1723.35 alert=retest2 |

### Cycle 3 — SELL (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 10:15:00 | 1441.03 | 1733.38 | 1734.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 13:15:00 | 1432.28 | 1724.59 | 1730.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1591.67 | 1557.28 | 1628.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 1591.67 | 1557.28 | 1628.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1602.87 | 1558.18 | 1628.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 12:30:00 | 1580.45 | 1559.21 | 1627.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 13:15:00 | 1501.43 | 1556.79 | 1621.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 1563.63 | 1555.64 | 1620.02 | SL hit (close>ema200) qty=0.50 sl=1555.64 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1848.23 | 1667.91 | 1667.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1862.07 | 1671.62 | 1668.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 2589.00 | 2590.17 | 2369.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 2575.00 | 2590.17 | 2369.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2469.50 | 2674.61 | 2498.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2469.50 | 2674.61 | 2498.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2408.70 | 2671.97 | 2498.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2408.70 | 2671.97 | 2498.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 2474.40 | 2666.18 | 2498.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 2491.00 | 2666.18 | 2498.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 2419.60 | 2641.15 | 2497.96 | SL hit (close<static) qty=1.00 sl=2455.10 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 2306.30 | 2465.00 | 2465.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 2290.00 | 2460.08 | 2462.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2366.50 | 2354.16 | 2400.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 2351.20 | 2354.16 | 2400.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2277.00 | 2201.37 | 2274.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2277.00 | 2201.37 | 2274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2318.00 | 2202.53 | 2274.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 2318.00 | 2202.53 | 2274.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2325.00 | 2203.75 | 2274.93 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 2486.50 | 2325.56 | 2325.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2500.00 | 2327.29 | 2326.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 2686.50 | 2734.65 | 2611.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-09 09:45:00 | 2669.70 | 2734.65 | 2611.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2579.90 | 2729.09 | 2615.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 2579.90 | 2729.09 | 2615.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 2582.10 | 2727.63 | 2615.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 2648.10 | 2727.63 | 2615.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 2602.00 | 2717.67 | 2622.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 2595.00 | 2695.00 | 2636.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-13 09:15:00 | 2862.20 | 2696.68 | 2652.58 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 09:15:00 | 878.15 | 2024-05-29 14:15:00 | 877.88 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-05-23 14:00:00 | 871.67 | 2024-05-29 14:15:00 | 877.88 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-05-23 14:30:00 | 870.98 | 2024-05-30 10:15:00 | 874.37 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-05-23 15:15:00 | 874.33 | 2024-05-30 13:15:00 | 854.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-05-24 12:15:00 | 883.33 | 2024-05-30 13:15:00 | 854.00 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-05-29 13:30:00 | 882.33 | 2024-05-30 13:15:00 | 854.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-05-30 09:15:00 | 884.70 | 2024-05-30 13:15:00 | 854.00 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2024-05-31 14:45:00 | 890.03 | 2024-06-04 10:15:00 | 822.40 | STOP_HIT | 1.00 | -7.60% |
| SELL | retest2 | 2024-07-31 15:00:00 | 851.50 | 2024-08-01 09:15:00 | 868.68 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-08-02 14:30:00 | 856.33 | 2024-08-05 09:15:00 | 813.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 14:30:00 | 856.33 | 2024-08-06 14:15:00 | 770.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-12 14:45:00 | 856.72 | 2024-08-16 12:15:00 | 874.53 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-08-13 10:00:00 | 853.30 | 2024-08-16 12:15:00 | 874.53 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-01-28 13:00:00 | 1765.00 | 2025-01-28 14:15:00 | 1720.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-01-29 09:30:00 | 1774.03 | 2025-02-05 14:15:00 | 1941.86 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2025-01-29 10:00:00 | 1765.33 | 2025-02-05 15:15:00 | 1951.43 | TARGET_HIT | 1.00 | 10.54% |
| BUY | retest2 | 2025-01-29 14:00:00 | 1774.62 | 2025-02-06 09:15:00 | 1952.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-12 12:45:00 | 1797.15 | 2025-02-13 14:15:00 | 1749.37 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-02-13 11:45:00 | 1789.87 | 2025-02-13 14:15:00 | 1749.37 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-02-19 09:30:00 | 1815.78 | 2025-02-21 09:15:00 | 1997.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-24 12:30:00 | 1580.45 | 2025-03-26 13:15:00 | 1501.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 12:30:00 | 1580.45 | 2025-03-27 09:15:00 | 1563.63 | STOP_HIT | 0.50 | 1.06% |
| BUY | retest2 | 2025-07-08 15:15:00 | 2491.00 | 2025-07-11 09:15:00 | 2419.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-07-15 11:45:00 | 2481.80 | 2025-07-18 09:15:00 | 2454.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-17 12:15:00 | 2481.30 | 2025-07-18 09:15:00 | 2454.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-21 09:30:00 | 2502.20 | 2025-07-25 14:15:00 | 2452.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-12-11 09:15:00 | 2648.10 | 2026-01-13 09:15:00 | 2862.20 | TARGET_HIT | 1.00 | 8.09% |
| BUY | retest2 | 2025-12-16 10:30:00 | 2602.00 | 2026-01-13 09:15:00 | 2854.50 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2025-12-30 15:15:00 | 2595.00 | 2026-02-01 11:15:00 | 2550.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-02-01 12:45:00 | 2612.80 | 2026-02-01 13:15:00 | 2573.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-02-02 14:45:00 | 2689.80 | 2026-02-09 09:15:00 | 2958.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-05 09:30:00 | 2681.00 | 2026-03-17 12:15:00 | 2949.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 09:30:00 | 2706.20 | 2026-03-17 13:15:00 | 2976.82 | TARGET_HIT | 1.00 | 10.00% |
