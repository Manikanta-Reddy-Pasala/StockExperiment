# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3602.30
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
| ALERT2_SKIP | 3 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 5 |
| TARGET_HIT | 16 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 18
- **Target hits / Stop hits / Partials:** 16 / 24 / 5
- **Avg / median % per leg:** 3.14% / 1.47%
- **Sum % (uncompounded):** 141.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 16 | 53.3% | 13 | 17 | 0 | 3.05% | 91.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 16 | 53.3% | 13 | 17 | 0 | 3.05% | 91.4% |
| SELL (all) | 15 | 11 | 73.3% | 3 | 7 | 5 | 3.33% | 50.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 11 | 73.3% | 3 | 7 | 5 | 3.33% | 50.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 27 | 60.0% | 16 | 24 | 5 | 3.14% | 141.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 1853.50 | 1944.55 | 1944.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 1849.50 | 1942.68 | 1943.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1891.48 | 1860.15 | 1892.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1891.48 | 1860.15 | 1892.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1891.48 | 1860.15 | 1892.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 1894.50 | 1860.15 | 1892.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1881.08 | 1861.08 | 1892.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 11:30:00 | 1870.40 | 1861.29 | 1891.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 11:30:00 | 1870.00 | 1861.70 | 1890.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:15:00 | 1776.88 | 1858.87 | 1888.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:15:00 | 1776.50 | 1858.87 | 1888.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 1857.50 | 1855.46 | 1886.05 | SL hit (close>ema200) qty=0.50 sl=1855.46 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1959.63 | 1901.68 | 1901.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 1971.45 | 1903.50 | 1902.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 1912.58 | 1915.83 | 1909.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 12:15:00 | 1912.58 | 1915.83 | 1909.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1912.58 | 1915.83 | 1909.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1912.58 | 1915.83 | 1909.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1901.00 | 1915.69 | 1909.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 1901.00 | 1915.69 | 1909.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1905.00 | 1915.58 | 1909.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 1950.00 | 1915.58 | 1909.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 1912.90 | 1915.82 | 1909.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:00:00 | 1916.98 | 1915.46 | 1909.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 14:00:00 | 1920.80 | 1918.10 | 1911.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1933.00 | 1925.82 | 1916.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 2054.50 | 1935.25 | 1923.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-11 09:15:00 | 2104.19 | 1950.61 | 1931.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 1888.75 | 2022.23 | 2022.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 1883.60 | 2000.02 | 2009.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 1908.68 | 1901.45 | 1948.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 10:45:00 | 1904.68 | 1901.45 | 1948.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1929.98 | 1868.54 | 1914.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:30:00 | 1911.20 | 1868.54 | 1914.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1923.93 | 1869.09 | 1914.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:30:00 | 1890.90 | 1927.59 | 1935.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 12:30:00 | 1900.05 | 1927.16 | 1935.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 1900.45 | 1926.84 | 1935.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:15:00 | 1805.05 | 1914.76 | 1928.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:15:00 | 1805.43 | 1914.76 | 1928.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 10:15:00 | 1796.36 | 1913.43 | 1927.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-12 10:15:00 | 1710.40 | 1893.22 | 1916.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1880.00 | 1813.78 | 1813.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1892.90 | 1814.57 | 1814.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 2876.80 | 2893.88 | 2709.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:45:00 | 2865.60 | 2893.88 | 2709.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2945.00 | 3052.15 | 2955.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2945.00 | 3052.15 | 2955.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2948.90 | 3051.12 | 2955.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 2960.90 | 3051.12 | 2955.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2928.90 | 3047.60 | 2955.00 | SL hit (close<static) qty=1.00 sl=2932.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 2956.40 | 2986.92 | 2987.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 2942.00 | 2986.48 | 2986.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 2977.80 | 2977.71 | 2982.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2994.00 | 2977.87 | 2982.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 2994.00 | 2977.87 | 2982.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 3033.00 | 2978.42 | 2982.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 3033.00 | 2978.42 | 2982.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2991.00 | 2979.45 | 2982.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:45:00 | 2990.20 | 2979.45 | 2982.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 2984.20 | 2979.50 | 2982.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 2969.50 | 2979.59 | 2982.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 15:15:00 | 2991.60 | 2979.71 | 2982.80 | SL hit (close>static) qty=1.00 sl=2987.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 3033.90 | 2985.76 | 2985.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 3055.00 | 2990.67 | 2988.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 3064.00 | 3066.80 | 3034.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 15:00:00 | 3064.00 | 3066.80 | 3034.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3011.40 | 3066.13 | 3034.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 3078.90 | 3063.61 | 3035.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:30:00 | 3081.60 | 3064.14 | 3036.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 2914.00 | 3060.29 | 3036.06 | SL hit (close<static) qty=1.00 sl=2992.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-22 11:30:00 | 1870.40 | 2024-08-26 09:15:00 | 1776.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-23 11:30:00 | 1870.00 | 2024-08-26 09:15:00 | 1776.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-22 11:30:00 | 1870.40 | 2024-08-27 09:15:00 | 1857.50 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2024-08-23 11:30:00 | 1870.00 | 2024-08-27 09:15:00 | 1857.50 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2024-08-27 14:30:00 | 1867.60 | 2024-08-30 13:15:00 | 1909.78 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-28 09:30:00 | 1868.90 | 2024-08-30 13:15:00 | 1909.78 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-09-19 15:15:00 | 1950.00 | 2024-10-11 09:15:00 | 2104.19 | TARGET_HIT | 1.00 | 7.91% |
| BUY | retest2 | 2024-09-20 11:15:00 | 1912.90 | 2024-10-11 09:15:00 | 2108.68 | TARGET_HIT | 1.00 | 10.23% |
| BUY | retest2 | 2024-09-20 14:00:00 | 1916.98 | 2024-10-11 09:15:00 | 2112.88 | TARGET_HIT | 1.00 | 10.22% |
| BUY | retest2 | 2024-09-24 14:00:00 | 1920.80 | 2024-10-18 12:15:00 | 2145.00 | TARGET_HIT | 1.00 | 11.67% |
| BUY | retest2 | 2024-10-09 09:15:00 | 2054.50 | 2024-10-30 14:15:00 | 1950.05 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2024-10-25 11:45:00 | 2003.55 | 2024-11-12 11:15:00 | 1966.23 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-25 14:15:00 | 1999.58 | 2024-11-12 11:15:00 | 1966.23 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-10-25 14:45:00 | 1999.15 | 2024-11-12 11:15:00 | 1966.23 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-10-30 09:15:00 | 2027.98 | 2024-11-12 12:15:00 | 1952.80 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-10-31 09:15:00 | 2003.33 | 2024-11-12 12:15:00 | 1952.80 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-10-31 11:45:00 | 2007.70 | 2024-11-12 12:15:00 | 1952.80 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-11-06 13:30:00 | 2001.60 | 2024-12-05 14:15:00 | 2199.54 | TARGET_HIT | 1.00 | 9.89% |
| BUY | retest2 | 2024-11-11 09:45:00 | 1981.00 | 2024-12-05 14:15:00 | 2199.07 | TARGET_HIT | 1.00 | 11.01% |
| BUY | retest2 | 2024-11-11 14:00:00 | 1979.05 | 2024-12-05 14:15:00 | 2195.66 | TARGET_HIT | 1.00 | 10.94% |
| BUY | retest2 | 2024-11-12 09:15:00 | 1989.58 | 2024-12-05 15:15:00 | 2203.91 | TARGET_HIT | 1.00 | 10.77% |
| BUY | retest2 | 2024-11-18 13:00:00 | 1996.05 | 2024-12-05 15:15:00 | 2211.72 | TARGET_HIT | 1.00 | 10.80% |
| BUY | retest2 | 2024-11-19 10:45:00 | 2010.65 | 2024-12-05 15:15:00 | 2202.03 | TARGET_HIT | 1.00 | 9.52% |
| BUY | retest2 | 2024-11-25 09:30:00 | 2001.85 | 2024-12-05 15:15:00 | 2206.58 | TARGET_HIT | 1.00 | 10.23% |
| BUY | retest2 | 2024-11-25 10:15:00 | 2005.98 | 2024-12-09 10:15:00 | 2259.95 | TARGET_HIT | 1.00 | 12.66% |
| BUY | retest2 | 2024-12-23 09:30:00 | 2002.68 | 2024-12-23 14:15:00 | 1966.03 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-03-05 10:30:00 | 1890.90 | 2025-03-10 09:15:00 | 1805.05 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2025-03-05 12:30:00 | 1900.05 | 2025-03-10 09:15:00 | 1805.43 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-03-05 13:30:00 | 1900.45 | 2025-03-10 10:15:00 | 1796.36 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-03-05 10:30:00 | 1890.90 | 2025-03-12 10:15:00 | 1710.40 | TARGET_HIT | 0.50 | 9.55% |
| SELL | retest2 | 2025-03-05 12:30:00 | 1900.05 | 2025-03-12 11:15:00 | 1701.81 | TARGET_HIT | 0.50 | 10.43% |
| SELL | retest2 | 2025-03-05 13:30:00 | 1900.45 | 2025-03-12 11:15:00 | 1710.05 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-05-28 13:45:00 | 1896.70 | 2025-05-29 15:15:00 | 1880.00 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-11-19 09:15:00 | 2960.90 | 2025-11-19 11:15:00 | 2928.90 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-15 09:15:00 | 2962.40 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-12-15 10:45:00 | 2961.40 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-12-15 14:00:00 | 2955.70 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2026-01-21 13:15:00 | 3038.20 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2026-01-21 14:30:00 | 3035.20 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2026-01-21 15:00:00 | 3045.10 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2026-02-12 14:30:00 | 2969.50 | 2026-02-12 15:15:00 | 2991.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2930.10 | 2026-02-13 12:15:00 | 2989.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-03-17 15:15:00 | 3078.90 | 2026-03-20 09:15:00 | 2914.00 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest2 | 2026-03-18 10:30:00 | 3081.60 | 2026-03-20 09:15:00 | 2914.00 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3101.20 | 2026-04-09 09:15:00 | 3411.32 | TARGET_HIT | 1.00 | 10.00% |
