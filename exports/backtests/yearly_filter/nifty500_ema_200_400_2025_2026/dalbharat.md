# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1840.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** -1.37% / -1.39%
- **Sum % (uncompounded):** -17.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.38% | -5.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.38% | -5.5% |
| SELL (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -1.36% | -12.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 8 | 1 | -1.36% | -12.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 1 | 7.7% | 0 | 12 | 1 | -1.37% | -17.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2184.00 | 2270.17 | 2270.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 2150.00 | 2268.98 | 2269.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2049.30 | 2030.56 | 2093.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 2056.00 | 2030.56 | 2093.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2090.20 | 2034.30 | 2092.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 2091.40 | 2034.30 | 2092.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2093.80 | 2034.89 | 2092.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 2081.90 | 2036.90 | 2092.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 2104.70 | 2038.65 | 2082.97 | SL hit (close>static) qty=1.00 sl=2101.90 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 15:15:00 | 2179.70 | 2105.61 | 2105.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 09:15:00 | 2198.80 | 2106.54 | 2105.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 2114.50 | 2116.62 | 2111.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 2133.00 | 2116.78 | 2111.28 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2048.30 | 2106.44 | 2106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2034.40 | 2105.47 | 2106.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:45:00 | 2102.50 | 2097.89 | 2102.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 2103.60 | 2097.95 | 2102.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:00:00 | 2103.60 | 2097.95 | 2102.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 2111.20 | 2098.08 | 2102.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:45:00 | 2106.00 | 2098.08 | 2102.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 2110.10 | 2098.20 | 2102.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:15:00 | 2105.70 | 2098.20 | 2102.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2105.70 | 2098.27 | 2102.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2136.30 | 2098.27 | 2102.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2133.40 | 2098.62 | 2102.43 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2162.10 | 2106.32 | 2106.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2176.80 | 2107.02 | 2106.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 2116.10 | 2125.00 | 2116.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 2115.00 | 2124.90 | 2116.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 2107.00 | 2124.90 | 2116.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 2116.10 | 2124.81 | 2116.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 2124.00 | 2124.81 | 2116.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2124.80 | 2124.63 | 2116.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 2131.90 | 2124.70 | 2116.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 2128.90 | 2124.79 | 2117.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 2129.90 | 2124.79 | 2117.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 2127.90 | 2124.91 | 2117.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 2107.50 | 2124.76 | 2117.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 2107.50 | 2124.76 | 2117.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 2100.20 | 2124.51 | 2117.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 2100.20 | 2124.51 | 2117.06 | SL hit (close<static) qty=1.00 sl=2106.20 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2046.80 | 2110.31 | 2110.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2019.40 | 2106.15 | 2108.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1943.20 | 1889.82 | 1959.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 1943.20 | 1889.82 | 1959.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1947.00 | 1894.04 | 1956.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1920.00 | 1897.30 | 1956.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 1915.00 | 1897.47 | 1956.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1985.90 | 1901.11 | 1956.59 | SL hit (close>static) qty=1.00 sl=1957.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-16 09:15:00 | 2081.90 | 2025-12-24 09:15:00 | 2104.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-08 11:00:00 | 2082.80 | 2026-01-13 12:15:00 | 2103.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-13 10:30:00 | 2083.90 | 2026-01-13 12:15:00 | 2103.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-18 13:00:00 | 2131.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-18 14:45:00 | 2128.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-19 09:15:00 | 2129.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-02-19 09:45:00 | 2127.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1920.00 | 2026-04-15 09:15:00 | 1985.90 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-04-13 10:00:00 | 1915.00 | 2026-04-15 09:15:00 | 1985.90 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-24 11:00:00 | 1924.90 | 2026-04-24 14:15:00 | 1961.30 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-04-28 13:30:00 | 1925.40 | 2026-04-29 10:15:00 | 1964.30 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1927.50 | 2026-05-04 10:15:00 | 1989.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-05-08 10:00:00 | 1918.00 | 2026-05-08 14:15:00 | 1822.10 | PARTIAL | 0.50 | 5.00% |
