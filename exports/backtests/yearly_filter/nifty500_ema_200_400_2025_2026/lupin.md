# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2373.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 19
- **Target hits / Stop hits / Partials:** 2 / 28 / 6
- **Avg / median % per leg:** 0.93% / -0.74%
- **Sum % (uncompounded):** 33.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 7 | 0 | 1.70% | 15.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 5 | 55.6% | 2 | 7 | 0 | 1.70% | 15.3% |
| SELL (all) | 27 | 12 | 44.4% | 0 | 21 | 6 | 0.68% | 18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 12 | 44.4% | 0 | 21 | 6 | 0.68% | 18.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 17 | 47.2% | 2 | 28 | 6 | 0.93% | 33.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 2073.10 | 2043.73 | 2043.59 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1985.60 | 2043.56 | 2043.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1977.90 | 2042.91 | 2043.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1998.10 | 1998.10 | 2016.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 1998.10 | 1998.10 | 2016.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2004.80 | 1997.98 | 2014.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 2001.20 | 1997.98 | 2014.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 14:15:00 | 2002.20 | 1998.06 | 2014.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:15:00 | 2001.50 | 1998.47 | 2014.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 2024.40 | 1998.96 | 2014.48 | SL hit (close>static) qty=1.00 sl=2017.60 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 2040.50 | 1953.56 | 1953.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 2047.30 | 1954.50 | 1953.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1969.70 | 1983.67 | 1970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1962.80 | 1983.46 | 1970.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1962.80 | 1983.46 | 1970.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1963.00 | 1983.26 | 1970.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1918.40 | 1983.26 | 1970.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1973.69 | 1967.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1933.80 | 1973.69 | 1967.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1938.10 | 1973.34 | 1967.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1971.00 | 1964.56 | 1963.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1944.20 | 1964.37 | 1963.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1948.30 | 1964.21 | 1963.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:45:00 | 1944.80 | 1962.42 | 1962.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1949.90 | 1962.12 | 1962.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 11:15:00 | 1949.90 | 1962.12 | 1962.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1941.10 | 1961.75 | 1961.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 1954.20 | 1950.56 | 1955.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1953.00 | 1950.59 | 1955.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:15:00 | 1957.50 | 1950.59 | 1955.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1953.50 | 1950.62 | 1955.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1940.00 | 1950.67 | 1955.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 1970.00 | 1950.08 | 1955.14 | SL hit (close>static) qty=1.00 sl=1960.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2020.00 | 1959.81 | 1959.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2034.30 | 1964.82 | 1962.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1991.60 | 1995.52 | 1980.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1991.60 | 1995.52 | 1980.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2098.90 | 2135.69 | 2099.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 2105.60 | 2135.69 | 2099.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2103.40 | 2135.37 | 2099.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2114.10 | 2135.37 | 2099.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 2112.10 | 2135.09 | 2101.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 2119.10 | 2134.82 | 2101.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 2085.30 | 2134.11 | 2102.13 | SL hit (close<static) qty=1.00 sl=2088.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:15:00 | 1979.50 | 2025-05-12 14:15:00 | 2042.60 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-06-10 13:15:00 | 2001.20 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-10 14:15:00 | 2002.20 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-11 11:15:00 | 2001.50 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2000.10 | 2025-07-10 09:15:00 | 1900.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2000.10 | 2025-07-15 11:15:00 | 1951.50 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-06-17 10:45:00 | 1966.00 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-02 10:00:00 | 1966.40 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-02 12:15:00 | 1971.10 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1971.30 | 2025-08-01 11:15:00 | 1872.54 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1951.40 | 2025-08-01 11:15:00 | 1872.73 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1957.90 | 2025-08-01 14:15:00 | 1867.70 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-07-29 09:15:00 | 1953.60 | 2025-08-01 14:15:00 | 1868.08 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1957.60 | 2025-08-04 09:15:00 | 1859.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1971.30 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1951.40 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1957.90 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2025-07-29 09:15:00 | 1953.60 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1957.60 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1921.70 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 11:00:00 | 1936.00 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-12 09:30:00 | 1933.80 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-20 13:00:00 | 1935.00 | 2025-08-21 10:15:00 | 1957.70 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-05 12:30:00 | 1935.10 | 2025-09-08 12:15:00 | 1955.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1936.00 | 2025-09-08 12:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1971.00 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-10 09:30:00 | 1944.20 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1948.30 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-10-15 09:45:00 | 1944.80 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1940.00 | 2025-10-31 10:15:00 | 1970.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-03 10:45:00 | 1951.80 | 2025-11-03 11:15:00 | 1990.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-29 11:15:00 | 2114.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-01 12:30:00 | 2112.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-01 14:15:00 | 2119.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-02 14:30:00 | 2111.80 | 2026-02-26 09:15:00 | 2322.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 2238.30 | 2026-05-07 09:15:00 | 2462.13 | TARGET_HIT | 1.00 | 10.00% |
