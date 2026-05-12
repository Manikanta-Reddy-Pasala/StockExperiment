# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3043.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 17 |
| TARGET_HIT | 16 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 25
- **Target hits / Stop hits / Partials:** 16 / 32 / 17
- **Avg / median % per leg:** 2.74% / 5.00%
- **Sum % (uncompounded):** 178.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 6 | 25.0% | 6 | 18 | 0 | 0.43% | 10.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 6 | 25.0% | 6 | 18 | 0 | 0.43% | 10.3% |
| SELL (all) | 41 | 34 | 82.9% | 10 | 14 | 17 | 4.09% | 167.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 34 | 82.9% | 10 | 14 | 17 | 4.09% | 167.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 40 | 61.5% | 16 | 32 | 17 | 2.74% | 178.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 811.50 | 846.18 | 846.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 802.00 | 844.52 | 845.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 847.10 | 843.70 | 844.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 847.10 | 843.70 | 844.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 847.10 | 843.70 | 844.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:45:00 | 845.60 | 843.70 | 844.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 832.50 | 843.59 | 844.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 12:00:00 | 825.00 | 843.40 | 844.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 12:30:00 | 825.00 | 843.09 | 844.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 10:15:00 | 823.65 | 837.87 | 841.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:45:00 | 822.85 | 837.61 | 841.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 831.45 | 836.87 | 841.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 12:00:00 | 827.00 | 836.70 | 840.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 826.75 | 836.43 | 840.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 783.75 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 783.75 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 782.47 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 781.71 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 785.65 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 785.41 | 830.21 | 837.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 742.50 | 825.94 | 834.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 15:15:00 | 895.55 | 825.02 | 824.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 915.35 | 845.84 | 836.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 12:15:00 | 910.40 | 913.05 | 878.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 12:45:00 | 915.00 | 913.05 | 878.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 900.00 | 914.60 | 882.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 915.90 | 914.60 | 882.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:15:00 | 906.00 | 914.23 | 883.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:15:00 | 905.10 | 914.12 | 883.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 13:45:00 | 906.50 | 913.94 | 883.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-16 14:15:00 | 995.61 | 924.85 | 892.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 1786.60 | 1906.46 | 1906.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 15:15:00 | 1774.20 | 1900.50 | 1903.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 1858.60 | 1856.32 | 1879.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 13:00:00 | 1858.60 | 1856.32 | 1879.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1815.85 | 1751.67 | 1804.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1815.85 | 1751.67 | 1804.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1812.25 | 1752.27 | 1804.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 1796.95 | 1752.27 | 1804.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:45:00 | 1799.00 | 1752.66 | 1804.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1707.10 | 1753.88 | 1800.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1709.05 | 1753.88 | 1800.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 1768.25 | 1754.02 | 1800.33 | SL hit (close>ema200) qty=0.50 sl=1754.02 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1720.00 | 1487.18 | 1486.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1742.55 | 1533.82 | 1511.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1463.95 | 1550.37 | 1521.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1463.95 | 1550.37 | 1521.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1463.95 | 1550.37 | 1521.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 1538.00 | 1547.42 | 1520.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 10:45:00 | 1529.90 | 1547.69 | 1522.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-15 09:15:00 | 1691.80 | 1554.43 | 1527.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2442.30 | 2624.58 | 2625.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 2432.20 | 2619.19 | 2622.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 2532.10 | 2506.86 | 2554.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 2532.10 | 2506.86 | 2554.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 2537.00 | 2507.16 | 2554.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 2549.80 | 2507.16 | 2554.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2586.40 | 2508.22 | 2554.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 2586.40 | 2508.22 | 2554.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2586.40 | 2509.00 | 2554.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 2574.30 | 2509.00 | 2554.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2561.00 | 2505.81 | 2549.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 2578.80 | 2505.81 | 2549.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2538.00 | 2506.13 | 2549.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 2509.50 | 2572.79 | 2576.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 2518.20 | 2569.35 | 2574.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 2510.50 | 2568.75 | 2574.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 2517.00 | 2568.23 | 2574.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2615.10 | 2567.44 | 2573.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 2615.10 | 2567.44 | 2573.64 | SL hit (close>static) qty=1.00 sl=2560.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 2724.70 | 2579.87 | 2579.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 2749.70 | 2588.04 | 2583.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 2607.00 | 2618.17 | 2600.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 2607.70 | 2618.17 | 2600.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 2604.00 | 2617.76 | 2601.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2624.30 | 2617.76 | 2601.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 2597.40 | 2617.56 | 2601.03 | SL hit (close<static) qty=1.00 sl=2600.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 2545.90 | 2593.83 | 2594.06 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2728.10 | 2594.75 | 2594.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 2780.00 | 2601.25 | 2597.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 2715.00 | 2718.74 | 2667.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 2715.00 | 2718.74 | 2667.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2654.00 | 2732.53 | 2683.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 2654.00 | 2732.53 | 2683.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2622.10 | 2731.43 | 2683.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 2622.10 | 2731.43 | 2683.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 2405.60 | 2646.31 | 2646.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 2398.20 | 2643.84 | 2645.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2496.20 | 2492.52 | 2554.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:30:00 | 2500.80 | 2492.52 | 2554.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2553.70 | 2493.38 | 2552.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 2558.20 | 2493.38 | 2552.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 2549.80 | 2493.94 | 2552.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 2554.60 | 2493.94 | 2552.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2538.90 | 2481.89 | 2534.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2538.90 | 2481.89 | 2534.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2525.00 | 2482.32 | 2534.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:30:00 | 2519.50 | 2482.69 | 2534.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 2519.50 | 2482.69 | 2534.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:30:00 | 2513.90 | 2482.79 | 2534.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 2514.00 | 2484.38 | 2532.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2507.70 | 2484.37 | 2530.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 2535.70 | 2484.37 | 2530.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2393.53 | 2483.25 | 2528.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2393.53 | 2483.25 | 2528.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2388.20 | 2483.25 | 2528.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2388.30 | 2483.25 | 2528.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 2488.30 | 2482.36 | 2526.99 | SL hit (close>ema200) qty=0.50 sl=2482.36 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 2676.90 | 2424.62 | 2423.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 2745.30 | 2432.54 | 2427.96 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-02-28 12:00:00 | 825.00 | 2024-03-12 09:15:00 | 783.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 12:30:00 | 825.00 | 2024-03-12 09:15:00 | 783.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 10:15:00 | 823.65 | 2024-03-12 09:15:00 | 782.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 11:45:00 | 822.85 | 2024-03-12 09:15:00 | 781.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 12:00:00 | 827.00 | 2024-03-12 09:15:00 | 785.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:15:00 | 826.75 | 2024-03-12 09:15:00 | 785.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 12:00:00 | 825.00 | 2024-03-13 09:15:00 | 742.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-28 12:30:00 | 825.00 | 2024-03-13 09:15:00 | 742.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-04 10:15:00 | 823.65 | 2024-03-13 09:15:00 | 741.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-04 11:45:00 | 822.85 | 2024-03-13 09:15:00 | 740.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 12:00:00 | 827.00 | 2024-03-13 09:15:00 | 744.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-06 09:15:00 | 826.75 | 2024-03-13 09:15:00 | 744.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-10 09:15:00 | 915.90 | 2024-05-16 14:15:00 | 995.61 | TARGET_HIT | 1.00 | 8.70% |
| BUY | retest2 | 2024-05-13 11:15:00 | 906.00 | 2024-05-17 09:15:00 | 1007.49 | TARGET_HIT | 1.00 | 11.20% |
| BUY | retest2 | 2024-05-13 12:15:00 | 905.10 | 2024-05-17 09:15:00 | 996.60 | TARGET_HIT | 1.00 | 10.11% |
| BUY | retest2 | 2024-05-13 13:45:00 | 906.50 | 2024-05-17 09:15:00 | 997.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 14:30:00 | 2215.00 | 2024-08-13 14:15:00 | 1942.00 | STOP_HIT | 1.00 | -12.33% |
| BUY | retest2 | 2024-08-08 11:15:00 | 2190.65 | 2024-08-13 14:15:00 | 1942.00 | STOP_HIT | 1.00 | -11.35% |
| SELL | retest2 | 2024-10-15 12:15:00 | 1796.95 | 2024-10-18 09:15:00 | 1707.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 12:45:00 | 1799.00 | 2024-10-18 09:15:00 | 1709.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 12:15:00 | 1796.95 | 2024-10-18 10:15:00 | 1768.25 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2024-10-15 12:45:00 | 1799.00 | 2024-10-18 10:15:00 | 1768.25 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2024-10-18 13:45:00 | 1800.00 | 2024-10-21 09:15:00 | 1846.85 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-10-21 15:00:00 | 1801.70 | 2024-10-22 10:15:00 | 1711.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1749.60 | 2024-10-22 11:15:00 | 1662.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:00:00 | 1801.70 | 2024-10-22 14:15:00 | 1621.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1749.60 | 2024-10-22 14:15:00 | 1574.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 13:00:00 | 1795.50 | 2024-12-13 12:15:00 | 1705.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 13:00:00 | 1795.50 | 2024-12-13 12:15:00 | 1718.80 | STOP_HIT | 0.50 | 4.27% |
| BUY | retest2 | 2025-04-07 15:15:00 | 1538.00 | 2025-04-15 09:15:00 | 1691.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 10:45:00 | 1529.90 | 2025-04-15 09:15:00 | 1682.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-29 12:00:00 | 2509.50 | 2025-10-01 09:15:00 | 2615.10 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-09-30 10:30:00 | 2518.20 | 2025-10-01 09:15:00 | 2615.10 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-09-30 11:30:00 | 2510.50 | 2025-10-01 09:15:00 | 2615.10 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-09-30 12:45:00 | 2517.00 | 2025-10-01 09:15:00 | 2615.10 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-10-14 09:15:00 | 2624.30 | 2025-10-14 09:15:00 | 2597.40 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-15 11:30:00 | 2610.00 | 2025-10-16 10:15:00 | 2599.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-15 12:15:00 | 2613.00 | 2025-10-16 10:15:00 | 2599.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-15 12:45:00 | 2609.90 | 2025-10-16 10:15:00 | 2599.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-16 14:00:00 | 2620.20 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-10-16 14:30:00 | 2624.00 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-10-17 09:30:00 | 2658.00 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2619.90 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-20 12:30:00 | 2637.00 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-10-20 15:15:00 | 2639.00 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-10-23 11:00:00 | 2645.00 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-10-23 12:30:00 | 2639.70 | 2025-10-27 09:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-10-28 09:15:00 | 2600.20 | 2025-10-28 13:15:00 | 2562.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-28 11:15:00 | 2580.00 | 2025-10-28 13:15:00 | 2562.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-30 11:00:00 | 2581.10 | 2025-10-30 11:15:00 | 2563.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-03 15:15:00 | 2580.00 | 2025-11-04 09:15:00 | 2565.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-06 12:30:00 | 2519.50 | 2026-01-12 09:15:00 | 2393.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:00:00 | 2519.50 | 2026-01-12 09:15:00 | 2393.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:30:00 | 2513.90 | 2026-01-12 09:15:00 | 2388.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 2514.00 | 2026-01-12 09:15:00 | 2388.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:30:00 | 2519.50 | 2026-01-12 15:15:00 | 2488.30 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2026-01-06 13:00:00 | 2519.50 | 2026-01-12 15:15:00 | 2488.30 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2026-01-06 13:30:00 | 2513.90 | 2026-01-12 15:15:00 | 2488.30 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2026-01-08 10:45:00 | 2514.00 | 2026-01-12 15:15:00 | 2488.30 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2026-02-06 13:00:00 | 2393.00 | 2026-02-09 13:15:00 | 2520.10 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2026-03-02 12:15:00 | 2393.00 | 2026-03-06 09:15:00 | 2518.30 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2026-03-13 09:15:00 | 2391.70 | 2026-03-16 09:15:00 | 2272.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2352.70 | 2026-03-23 09:15:00 | 2235.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 2391.70 | 2026-03-23 10:15:00 | 2152.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2352.70 | 2026-03-23 12:15:00 | 2117.43 | TARGET_HIT | 0.50 | 10.00% |
