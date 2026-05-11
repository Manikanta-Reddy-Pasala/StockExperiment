# Hyundai Motor India Ltd. (HYUNDAI)

## Backtest Summary

- **Window:** 2024-10-22 09:15:00 → 2026-05-08 15:15:00 (2664 bars)
- **Last close:** 1833.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 3
- **Target hits / Stop hits / Partials:** 4 / 11 / 12
- **Avg / median % per leg:** 4.36% / 5.00%
- **Sum % (uncompounded):** 117.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| SELL (all) | 26 | 24 | 92.3% | 4 | 10 | 12 | 4.67% | 121.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 24 | 92.3% | 4 | 10 | 12 | 4.67% | 121.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 24 | 88.9% | 4 | 11 | 12 | 4.36% | 117.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 13:15:00 | 1809.85 | 1797.97 | 1797.94 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 1796.90 | 1797.89 | 1797.90 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1832.95 | 1798.04 | 1797.97 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1708.15 | 1797.92 | 1797.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 1702.70 | 1796.08 | 1797.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 1721.50 | 1704.95 | 1740.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 1721.50 | 1704.95 | 1740.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1721.50 | 1704.95 | 1740.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 1726.40 | 1704.95 | 1740.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 1746.20 | 1705.85 | 1740.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:00:00 | 1746.20 | 1705.85 | 1740.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 1750.30 | 1706.29 | 1740.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:45:00 | 1749.05 | 1706.29 | 1740.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1746.05 | 1708.48 | 1740.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 1744.00 | 1708.48 | 1740.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1733.00 | 1708.72 | 1740.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 1751.80 | 1708.72 | 1740.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1745.45 | 1709.09 | 1740.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 1734.65 | 1709.09 | 1740.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 1735.25 | 1708.85 | 1738.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1647.92 | 1706.89 | 1734.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1648.49 | 1706.89 | 1734.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 1561.19 | 1698.46 | 1728.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 1830.10 | 1717.31 | 1717.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 1835.70 | 1719.60 | 1718.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 2059.60 | 2073.16 | 1996.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 2059.60 | 2073.16 | 1996.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2419.30 | 2539.10 | 2423.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 2419.30 | 2539.10 | 2423.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2398.20 | 2537.70 | 2423.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2398.20 | 2537.70 | 2423.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2408.90 | 2535.19 | 2423.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 2405.60 | 2535.19 | 2423.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2407.30 | 2532.83 | 2423.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 2406.60 | 2532.83 | 2423.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2409.50 | 2531.60 | 2423.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2402.50 | 2531.60 | 2423.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2390.20 | 2511.71 | 2421.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 2390.20 | 2511.71 | 2421.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2469.20 | 2504.78 | 2420.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 2470.50 | 2504.78 | 2420.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 2376.50 | 2499.68 | 2420.92 | SL hit (close<static) qty=1.00 sl=2406.80 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 2322.00 | 2384.35 | 2384.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 2314.50 | 2383.65 | 2384.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 2373.60 | 2367.19 | 2375.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2388.50 | 2367.40 | 2375.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 2387.40 | 2367.40 | 2375.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2399.20 | 2367.72 | 2375.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 2399.20 | 2367.72 | 2375.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 2386.70 | 2368.48 | 2375.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 2386.70 | 2368.48 | 2375.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2389.00 | 2368.68 | 2375.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 2384.10 | 2368.68 | 2375.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 2341.00 | 2368.50 | 2375.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 2323.70 | 2369.15 | 2375.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 2334.90 | 2349.57 | 2363.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 2319.30 | 2349.49 | 2363.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 2284.20 | 2347.94 | 2362.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2342.00 | 2313.05 | 2333.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 2342.00 | 2313.05 | 2333.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2334.30 | 2313.26 | 2333.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:15:00 | 2342.50 | 2313.26 | 2333.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2335.40 | 2313.48 | 2333.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 2312.00 | 2316.91 | 2334.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:30:00 | 2314.70 | 2310.92 | 2328.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 2291.10 | 2312.79 | 2328.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:45:00 | 2325.00 | 2312.96 | 2328.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2316.00 | 2312.99 | 2328.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:45:00 | 2328.60 | 2312.99 | 2328.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2207.51 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2218.16 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2203.34 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2169.99 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2196.40 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2198.96 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2176.54 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2208.75 | 2302.88 | 2321.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-26 10:15:00 | 1734.65 | 2025-04-03 09:15:00 | 1647.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 15:15:00 | 1735.25 | 2025-04-03 09:15:00 | 1648.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 10:15:00 | 1734.65 | 2025-04-07 09:15:00 | 1561.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 15:15:00 | 1735.25 | 2025-04-07 09:15:00 | 1561.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 14:30:00 | 1738.60 | 2025-05-07 13:15:00 | 1765.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-07 09:15:00 | 1723.00 | 2025-05-07 13:15:00 | 1765.70 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-10-15 11:15:00 | 2470.50 | 2025-10-16 10:15:00 | 2376.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-12-05 11:15:00 | 2323.70 | 2026-01-27 09:15:00 | 2207.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:45:00 | 2334.90 | 2026-01-27 09:15:00 | 2218.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2319.30 | 2026-01-27 09:15:00 | 2203.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 09:15:00 | 2284.20 | 2026-01-27 09:15:00 | 2169.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 2312.00 | 2026-01-27 09:15:00 | 2196.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2314.70 | 2026-01-27 09:15:00 | 2198.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 2291.10 | 2026-01-27 09:15:00 | 2176.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:45:00 | 2325.00 | 2026-01-27 09:15:00 | 2208.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:15:00 | 2323.70 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-12-11 14:45:00 | 2334.90 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2319.30 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-12-15 09:15:00 | 2284.20 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2026-01-09 09:15:00 | 2312.00 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2314.70 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-01-20 09:15:00 | 2291.10 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2026-01-20 13:45:00 | 2325.00 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-02-23 10:00:00 | 2242.10 | 2026-03-02 09:15:00 | 2129.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2244.40 | 2026-03-02 09:15:00 | 2132.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:00:00 | 2242.10 | 2026-03-09 09:15:00 | 2017.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2244.40 | 2026-03-09 09:15:00 | 2019.96 | TARGET_HIT | 0.50 | 10.00% |
