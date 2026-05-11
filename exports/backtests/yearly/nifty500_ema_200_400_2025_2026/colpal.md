# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (2707 bars)
- **Last close:** 2193.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 0 / 14 / 2
- **Avg / median % per leg:** -0.45% / -0.86%
- **Sum % (uncompounded):** -7.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.51% | -5.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.51% | -5.0% |
| SELL (all) | 14 | 4 | 28.6% | 0 | 12 | 2 | -0.15% | -2.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 0 | 12 | 2 | -0.15% | -2.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 4 | 25.0% | 0 | 14 | 2 | -0.45% | -7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 2619.50 | 2556.65 | 2556.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2642.10 | 2567.08 | 2561.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 2510.20 | 2594.43 | 2577.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 2510.20 | 2594.43 | 2577.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2510.20 | 2594.43 | 2577.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 2516.90 | 2594.43 | 2577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2524.10 | 2593.73 | 2577.30 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 14:15:00 | 2489.70 | 2563.04 | 2563.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 2471.70 | 2559.10 | 2561.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2441.00 | 2440.74 | 2479.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 2441.00 | 2440.74 | 2479.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2475.70 | 2441.70 | 2477.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 2479.50 | 2441.70 | 2477.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2480.70 | 2442.09 | 2477.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 2481.00 | 2442.09 | 2477.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 2479.20 | 2442.45 | 2477.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 2486.80 | 2442.45 | 2477.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 2469.70 | 2442.73 | 2477.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 2475.50 | 2442.73 | 2477.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 2174.50 | 2108.28 | 2162.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 2186.40 | 2108.28 | 2162.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 2176.20 | 2108.95 | 2162.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 2176.20 | 2108.95 | 2162.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 2160.60 | 2110.77 | 2162.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:15:00 | 2156.80 | 2110.77 | 2162.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 2156.50 | 2113.10 | 2160.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 2178.60 | 2113.75 | 2160.21 | SL hit (close>static) qty=1.00 sl=2176.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 2281.70 | 2163.10 | 2162.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 2290.50 | 2164.37 | 2163.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 2171.70 | 2176.43 | 2169.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 10:15:00 | 2171.70 | 2176.43 | 2169.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 2171.70 | 2176.43 | 2169.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 2171.70 | 2176.43 | 2169.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 2181.60 | 2176.48 | 2169.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 2191.40 | 2176.60 | 2170.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 2158.00 | 2178.90 | 2171.58 | SL hit (close<static) qty=1.00 sl=2167.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 1982.70 | 2165.20 | 2165.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 1975.50 | 2163.32 | 2164.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 1962.50 | 1956.43 | 2024.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:30:00 | 1963.20 | 1956.43 | 2024.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2098.80 | 1958.75 | 2023.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 2098.80 | 1958.75 | 2023.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 2105.50 | 1960.21 | 2024.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:30:00 | 2110.60 | 1960.21 | 2024.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 2163.00 | 2064.06 | 2064.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 2172.00 | 2068.87 | 2066.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-20 11:15:00 | 2156.80 | 2026-01-22 12:15:00 | 2178.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-01-22 11:30:00 | 2156.50 | 2026-01-22 12:15:00 | 2178.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-27 09:15:00 | 2151.50 | 2026-01-30 09:15:00 | 2043.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 09:15:00 | 2141.70 | 2026-01-30 09:15:00 | 2034.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 09:15:00 | 2151.50 | 2026-02-02 13:15:00 | 2118.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-01-28 09:15:00 | 2141.70 | 2026-02-02 13:15:00 | 2118.80 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2112.70 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-02-04 13:00:00 | 2115.00 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-02-04 15:15:00 | 2115.00 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-02-05 14:30:00 | 2114.60 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-02-09 14:30:00 | 2150.30 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-02-09 15:15:00 | 2152.00 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-12 10:15:00 | 2151.50 | 2026-02-18 09:15:00 | 2170.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-17 11:30:00 | 2154.00 | 2026-02-18 09:15:00 | 2170.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-03-05 12:45:00 | 2191.40 | 2026-03-09 09:15:00 | 2158.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-03-10 09:30:00 | 2188.40 | 2026-03-11 09:15:00 | 2112.00 | STOP_HIT | 1.00 | -3.49% |
